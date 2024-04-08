# [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/description/)
Given a string containing just the characters `'('` and `')'`, return the length of the longest valid (well-formed) parentheses substring.

## Solution (Beats 83%)
This technique employs dynamic programming to solve the problem by iterating over each character `l` in the string `s`. For each left bracket encountered, its index is saved in the `open_idx` stack. Simultaneously, during each iteration, we calculate the length of the longest valid parentheses substring ending precisely at `l`, recording this length in `prev_valid_l`. If `l` is a left bracket or `open_idx` is empty, we set the corresponding entry in `prev_valid_l` to `0`, indicating no valid substring ends at `l`. If `l` is a right bracket and `open_idx` is not empty, we remove the last index from `open_idx` to identify the matching opening bracket for `l`, and calculate the length of the current longest valid parentheses substring using this index.

## Code
```
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
        return max_l
```