# [Pow(x, n)](https://leetcode.com/problems/powx-n/?envType=company&envId=facebook&favoriteSlug=facebook-thirty-days)

## Description
Implement `pow(x, n)`, which calculates `x` raised to the power `n` (i.e., $x^n$).

## Hint
Power rules: $(x^m)^n=x^{mn}$ and $x^{m+n} = x^mx^n$.

## Solution (Beats 100%)
I wrote a recursive function to solve the power function:
- **Base case:** when `n` is 0, the function returns 1 ad when `n` is 1 the function returns `x`.
- **Negative exponent:**  The function converts the problem into a positive exponent using the power rule: $x^n = \frac{1}{x}^{-n}$. This ensures the recursion deals only with positive exponents.
- **Even Exponent:** For an even exponent, the function applies exponentiation by squaring: $x^n = (x*x)^{\frac{n}{2}}$
- **Odd Exponent:** For an odd exponent, one factor of $x$ is factored out, and the remaining exponent becomes even: $x^n=x(x*x)^{\frac{n-1}{2}}$.

```
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            # x^0 = 1
            return 1
        elif n == 1:
            # x^1 = x
            return x
        elif n < 0:
            # x^(-n) = 1/x^n
            return self.myPow(1/x, -n)
        elif n % 2 == 0:
            # x^(2n) = x^n * x^n
            return self.myPow(x*x, n//2)
        else:
            # x^(2n+1) = x^n * x^n * x
            return self.myPow(x, n-1)*x
```