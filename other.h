//
//  other.h
//  
//
//  Created by 王赫萌 on 2019/3/21.
//

#ifndef other_h
#define other_h
#include "common.h"
#define T int
//template<typename T>
void exclusive_scan(T *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    T old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

#endif /* other_h */
