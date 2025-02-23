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