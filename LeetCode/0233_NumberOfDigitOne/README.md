# [Number of Digit One](https://leetcode.com/problems/number-of-digit-one/)

Given an integer `n`, count the total number of digit `1` appearing in all non-negative integers less than or equal to `n`.

## Solution
TODO
## Code
```
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit_buff, tmp = 1, n
        sum_buff, results = 0, [0]
        while tmp > 0:
            tmp, curr_digit = divmod(tmp, 10)
            results.append(sum_buff*curr_digit+digit_buff if curr_digit>1 \
                           else sum_buff*curr_digit+(n%digit_buff+1)*curr_digit)
            sum_buff += sum_buff*9+digit_buff
            digit_buff *= 10
        return sum(results)
```