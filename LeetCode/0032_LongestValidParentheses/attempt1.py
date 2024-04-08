class Solution:
    def longestValidParentheses(self, s: str) -> int:
        open_idx, prev_valid_l, max_l= [], [0], 0
        for l_i, l in enumerate(s):
            if l == "(":
                open_idx.append(l_i)
                prev_valid_l.append(0)
            else:
                try:
                    p_i = open_idx.pop()
                    curr_valid_l = prev_valid_l[p_i]+l_i-p_i+1
                    prev_valid_l.append(curr_valid_l)
                    max_l = max(max_l, curr_valid_l)
                except:
                    prev_valid_l.append(0)
                    curr_valid_l = 0
        return max_l

if __name__ == "__main__":
    s = Solution()
    print(s.longestValidParentheses("(()"))
    print(s.longestValidParentheses(")()())"))
    print(s.longestValidParentheses(""))
    print(s.longestValidParentheses("()"))