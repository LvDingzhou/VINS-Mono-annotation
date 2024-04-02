#include "marginalization_factor.h"

/// @brief 待边缘化的各个残差块计算残差和雅可比矩阵，同时处理核函数
void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());//确定残差的维数

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();//确定参数块的数目
    raw_jacobians = new double *[block_sizes.size()];//new了给jacobi用的数组（这些都是一维数组）这里返回值是一个地址
    jacobians.resize(block_sizes.size());//雅可比的块数，vector的大小

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);//雅可比矩阵大小，残差x变量
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    //调用各自重载的接口计算残差和雅可比,用ceres来计算的
    // virtual bool Evaluate(double const* const* parameters,
                        // double* residuals,
                        // double** jacobians) const = 0;
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    if (loss_function)//如果有核函数
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();//获得残差的模
        loss_function->Evaluate(sq_norm, rho);//rho[0]:核函数这个点的值，rho[1]这个点的导数，rho[2]这个点的二阶段
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))//柯西核p=log(s+1),rho[2]<=0始终成立，一般核函数二阶导都是小于0
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;//都乘了一个尺度
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

/// @brief 收集各个残差
/// @param residual_block_info 
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);//残差块收集起来

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;//获得参数块的起始地址
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();//通过ceress的接口获得每个参数块大小

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)//遍历每个参数块
    {
        double *addr = parameter_blocks[i];//先拿到参数块的首地址
        int size = parameter_block_sizes[i];//拿到参数块的大小
        //**这里是一个unordered map（哈希表），避免重复添加**
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }
    
    //待边缘化的参数块
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        //先准备好待边缘化的参数块的unordered map
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}
/// @brief 将各个残差块计算残差和雅可比，同时备份所有相关参数
void MarginalizationInfo::preMarginalize()
{
    for (auto it : factors)
    {
        it->Evaluate();//调用这个接口计算各个残差块的残差和雅可比矩阵

        //得到每个残差块的参数块大小
        //这一步操作在`Evaluate()`中也做了
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();

        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);//得到该参数块的地址
            int size = block_sizes[i];
            //把各个参数块都备份起来，使用unorder map避免重复参数块，之所以备份是为了后面状态保留
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                //深拷贝
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;//地址->参数块实际内容的地址
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

//手写后端
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                //由jacobi算h阵J^T*J=H
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;//如果i=j,对角线上的一块儿
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();//这是i!=j的case
                }
            }
            //构建g矩阵，J^T*b;就是H*deltax=g,J^T*J=H,J^T*e=g
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

/// @brief 多线程构造先验项舒尔补AX=b的结构，计算Jacobian和残差
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    //需要边缘化的参数块
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);//计算每个参数块的地址
    }

    m = pos;//最终参数块的总大小

    //其他参数块
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;//每个参数块的大小都能被正确找到
            pos += localSize(it.second);
        }
    }

    n = pos - m;

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread

    //往A矩阵和b矩阵中填东西，利用多线程加速
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);//每个线程均匀分配任务
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        //等待各个线程完成各自的任务
        pthread_join( tids[i], NULL ); 
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }//这里H*deltax=g就摆好了，后面就可以舒尔补了
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    //TODO
    //这个操作保证了矩阵的正定性，因为在舒尔补中，H矩阵中的左上块儿是cap_lambda_a,这个cap_lambda_a是要进行逆操作的
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);//特征值分解

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());
    
    //(condition).select(A,B) == condition?A:B
    //利用特征值取逆来构造其逆矩阵
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    //这一块儿就是得到了舒尔补之后的方程形式，笔记21.1-cui(42)
    Eigen::VectorXd bmm = b.segment(0, m);//待边缘化大小g_a
    Eigen::MatrixXd Amr = A.block(0, m, m, n);//cap_lambda_b
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);//cap_lambda_b^T
    Eigen::MatrixXd Arr = A.block(m, m, n, n);//cap_lambda_c
    Eigen::VectorXd brr = b.segment(m, n);//g_b
    //构成了新的Ax=b,或者说是H'*deltax=g'
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    //这里面求H'*deltax=g'的jacobi和残差e
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    //这就是H'*deltax=g'新的jacobi和残差e
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);//留下来的参数块大小global size
            keep_block_idx.push_back(parameter_block_idx[it.first]);//留下来的原向量中的索引
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);//对应的新地址
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)//keep_block_size表示上一次边缘化留下来的参数块大小
    {
        mutable_parameter_block_sizes()->push_back(it);//调用ceres
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);//留下来的参数块大小
};

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n;// 保留的状态变量数目
    int m = marginalization_info->m;// 被边缘化的状态变量数目
    Eigen::VectorXd dx(n);//**dx表示 当前的状态量 和 边缘化时的状态量 之间的差值**
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;//idx起点 统一到0
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
