class Solution:
    def combine(self, n: int, k: int) -> list[list[int]]:

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

if __name__ == "__main__":
    sol = Solution()
    print(sol.combine(4, 2))
    print(sol.combine(1, 1))