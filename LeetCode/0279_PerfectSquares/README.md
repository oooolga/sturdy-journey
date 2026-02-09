# [Perfect Squares](https://leetcode.com/problems/perfect-squares/)

Given an integer `n`, return the least number of perfect square numbers that sum to `n`.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.

## Solution

TODO

## Code
```Py
class Solution:
    def numSquares(self, n: int) -> int:
        buff = 1
        curr_squares = [buff**2]
        sol = [1]

        for i in range(2, n+1, 1):
            if (buff+1)**2 == i:
                sol.append(1)
                buff += 1
                curr_squares.append(buff**2)
            else:
                least_squares = i
                for square in curr_squares:
                    if least_squares > sol[i-square-1] + 1:
                        least_squares = sol[i-square-1] + 1
                sol.append(least_squares)
        return sol[-1]
```