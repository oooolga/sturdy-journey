class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        factors, ret, ordered_elements = [1], "", []
        for i in range(1, n+1):
            factors.append(factors[-1]*i)
            ordered_elements.append(str(i))
        factors.pop()
        k = k-1
        for i in range(1, n+1):
            factor = factors.pop()
            element_idx, k = divmod(k, factor)
            ret += ordered_elements[element_idx]
            ordered_elements = ordered_elements[:element_idx] + ordered_elements[element_idx+1:]
        return ret

if __name__ == "__main__":
    print(Solution().getPermutation(3, 3))  # "213"
    print(Solution().getPermutation(4, 9))  # "2314"
    print(Solution().getPermutation(3, 1))  # "123"
    print(Solution().getPermutation(3, 2))  # "132"