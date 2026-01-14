#pragma once

#include <algorithm>
//aspin
// forward declaration（如果 PhysicalFieldT 已经在别处定义，可省）
template <typename T>
struct PhysicalFieldT;

inline void enforce_last_dof(PhysicalField &f)
{
    double sum = 0.0;
    for (int k = 0; k < DOF_reaction - 1; ++k)
        sum += f.cw[k];

    double last = 1.0 - sum;

    // 数值保护（建议保留）
    if (last < 0.0) last = 0.0;
    if (last > 1.0) last = 1.0;

    f.cw[DOF_reaction - 1] = last;
}


template <class T>
inline void enforce_last_dof_dual(PhysicalFieldT<T> &f)
{
    const int last = DOF_reaction - 1;
    T sum = T(0.0);
    for (int k = 0; k < last; ++k)
        sum += f.cw[k];

    // dual 里通常不 clamp，避免导数被截断（牛顿更稳）
    f.cw[last] = T(1.0) - sum;
}
