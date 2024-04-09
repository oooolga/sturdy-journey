# [Combinations](https://leetcode.com/problems/combinations/)
Given two integers `n` and `k`, return all possible combinations of `k` numbers chosen from the range `[1, n]`.

You may return the answer in **any order**.

## Solution (Beats 93%)
For each combination of `k` elements, if we arrange the elements of a given combination in ascending order, the range of possible values for the first element is `[1, n-k]`, for the second element is `[2, n-k+1]`, and so on, up to the `k`th element, which is bounded by `[k, n]`. We initialize `buff` with a combination where each element is at its minimum possible value. In each iteration, we increment the value of the last element in `buff` by 1. Should this increment take the element beyond its maximum allowable value, we then increment the value of the preceding element, maintaining the integrity of the combination within its defined bounds.

## Code
```
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:

        upper_bounds = list(range(n-k+1, n+1))
        # lower_bounds = list(range(1, k+1))
        buff = list(range(1, k+1))
        ret = [buff]

        def get_next(buff):
            buff = buff[:]
            tmp = -1
            # if buff==upper_bounds, the following while loop would raise an OutOfRange error
            while buff[tmp] == upper_bounds[tmp]:
                tmp -= 1
            buff[tmp] += 1
            if tmp != -1:
                tmp = tmp+1
                while tmp < 0:
                    buff[tmp] = buff[tmp-1]+1
                    tmp += 1
            return buff
        
        try:
            while True:
                buff = get_next(buff)
                if buff:
                    ret.append(buff)
        except:
            return ret
```