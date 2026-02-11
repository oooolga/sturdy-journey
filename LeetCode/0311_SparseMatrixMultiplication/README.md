# [Sparse Matrix Multiplication](https://leetcode.com/problems/sparse-matrix-multiplication/)
Given two sparse matrices `mat1` of size `m x k` and `mat2` of size `k x n`, return the result of `mat1 x mat2`. You may assume that multiplication is always possible.

## Solution
I would use two dictionaries, `mat1_D` and `mat2_D`, to store the non-zero entries of `mat1` (by row) and `mat2` (by column), respectively. Then, during multiplication, I only compute each output entry `(i, j)` when `mat1` has non-zeros in row `i` and `mat2` has non-zeros in column `j`, avoiding work on zero-only combinations.

## Code
```Py
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        mat1_D = {}
        k = len(mat1[0])
        for row_i, row in enumerate(mat1):
            row_dict = {}
            for item_i, item in enumerate(row):
                if item != 0:
                    row_dict[item_i] = item
            if len(row_dict.keys()):
                mat1_D[row_i] = row_dict
        mat2_D = {}
        for col_i in range(len(mat2[0])):
            col_dict = {}
            for row_i in range(len(mat2)):
                if mat2[row_i][col_i] != 0:
                    col_dict[row_i] = mat2[row_i][col_i]
            if len(col_dict.keys()):
                mat2_D[col_i] = col_dict
        
        ret = []
        for i in range(len(mat1)):
            ret_i = []
            for j in range(len(mat2[0])):
                if not i in mat1_D or not j in mat2_D:
                    ret_i.append(0)
                else:
                    buff = 0
                    for key in mat1_D[i]:
                        if key in mat2_D[j]:
                            buff += mat1_D[i][key] * mat2_D[j][key]
                    ret_i.append(buff)
            ret.append(ret_i)
        return ret
```