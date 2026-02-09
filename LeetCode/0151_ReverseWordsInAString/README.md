# [Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/description/)

Given an input string `s`, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

## Solution
Initialize two pointers, `A` and `B`, at the end of the string. Traverse the string backward with pointer `A` until a non-space character is found. Set pointer `B` to `A`'s current position and continue moving `B` leftward until it encounters a space. Extract the substring from `B + 1` to `A`, then resume the search with `A` from `B`'s current position.

## Code
```Python
class Solution:
    def reverseWords(self, s: str) -> str:
        pointerA = pointerB = 0
        ret = ''
        while pointerA > -len(s) and pointerB > -len(s):
            if pointerB <= pointerA:
                if s[pointerB-1] == ' ':
                    pointerB -= 1
                else:
                    pointerA = pointerB - 1
            else:
                if s[pointerA-1] != ' ':
                    pointerA -= 1
                else:
                    if ret:
                        ret += ' ' + s[len(s)+pointerA:len(s)+pointerB]
                    else:
                        ret = s[len(s)+pointerA:len(s)+pointerB]
                    pointerB = pointerA-1
        if pointerA < pointerB:
            if ret:
                ret += ' ' + s[len(s)+pointerA:len(s)+pointerB]
            else:
                ret = s[len(s)+pointerA:len(s)+pointerB]
        return ret
```
