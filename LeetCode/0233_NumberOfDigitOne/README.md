# [Number of Digit One](https://leetcode.com/problems/number-of-digit-one/)

Given an integer `n`, count the total number of digit `1` appearing in all non-negative integers less than or equal to `n`.

## Solution (Beats 93%)
We iterate over each digit of $n$, starting from the rightmost digit.

The maximum value for a number with $i$ digits is $10^i - 1$. Assuming we know $\text{countDigitOne}(10^i - 1)$ and aim to find $\text{countDigitOne}(10^{i+1} - 1)$, the formula is $\text{countDigitOne}(10^i - 1) \times 10 + 10^i$. Here, $\text{countDigitOne}(10^i - 1) \times 10$ represents the count of '1's in the last $i$ digits, and $10^i$ represents the count of '1's at the $i$-th digit.

For an $n$ with $k$ digits, $\text{countDigitOne}(n) = \text{countDigitOne}(10^{k-1} - 1) \times n_k + \min(10^{k-1}, n_{:k})$, where $n_k$ is the value of the $k$-th digit of $n$, and $n_{:k}$ is the value represented by the digits before the $k$-th digit.

In the following code, `sum_buff` keeps track of the values of $\text{countDigitOne}(10^i - 1)$.
## Code
```
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit_buff, tmp = 1, n
        sum_buff, result = 0, 0
        while tmp > 0:
            tmp, curr_digit = divmod(tmp, 10)
            result += sum_buff*curr_digit+digit_buff if curr_digit>1 \
                       else sum_buff*curr_digit+(n%digit_buff+1)*curr_digit
            sum_buff = sum_buff*10+digit_buff
            digit_buff *= 10
        return result
```