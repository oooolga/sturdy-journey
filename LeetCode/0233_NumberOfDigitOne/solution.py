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

if __name__ == "__main__":
    print(Solution().countDigitOne(13))
    print(Solution().countDigitOne(99))