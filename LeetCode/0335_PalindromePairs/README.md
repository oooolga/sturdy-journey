# [Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/description/)

You are given a 0-indexed array of unique strings `words`.

A **palindrome pair** is a pair of integers `(i, j)` such that:

- `0 <= i, j < words.length`,
- `i != j`, and
- `words[i] + words[j]` (the concatenation of the two strings) is a palindrome.
  
Return an array of all the **palindrome pairs** of `words`.

You must write an algorithm with `O(sum of words[i].length)` runtime complexity.

## Attempt 1
### High-level Idea

For any palindrome string `s`, there's a midpoint `m`. If `s` is even-length, `s[:m] == s[m:][::-1]`. If `s` is odd-length, `s[:m]==s[m+1:][::-1]`.

If `s` is formed by concatenating two strings, `s = words[i] + words[j]`, the midpoint is at `(len(words[i]) + len(words[j])) / 2`. When both strings are of equal length, the midpoint falls between them. If their lengths differ, the midpoint is within the longer string.

We examine each position in every word, treating it as the potential midpoint of a palindrome, to identify its palindrome counterpart.

### Code
```
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        # reverse_dict = {word[::-1]:word_i for word_i, word in enumerate(words)}
        reverse_idx = {}
        reverse_words = {}
        for word_i, word in enumerate(words):
            reversed_word = word[::-1]
            reverse_idx[reversed_word] = word_i
            reverse_words[word_i] = reversed_word

        def get_reversed_substring(word_i, i, j):
            i = -len(reverse_words[word_i]) if i==0 and j!=0 else i
            return reverse_words[word_i][-j:-i]

        ret = []

        def check_substrings(substring, substring_idx, if_start=True):
            if substring in reverse_idx and reverse_idx[substring] != substring_idx:
                if not if_start:
                    ret.append([substring_idx, reverse_idx[substring]])
                else:
                    ret.append([reverse_idx[substring], substring_idx])

        for word_j, word in enumerate(words):

            # midpoint at the start of word_j
            midpoint = 0
            left_half = "" # reversed
            try:
                while True:
                    left_half = get_reversed_substring(word_j, 0, midpoint)
                    right_half = word[midpoint:midpoint*2]
                    if left_half == right_half:
                        check_substrings(word[midpoint*2:], word_j)
                    
                    right_half = word[midpoint+1:midpoint*2+1]
                    if left_half == right_half:
                        check_substrings(word[midpoint*2+1:], word_j)
                    
                    midpoint += 1
                    assert midpoint <= len(word)//2
            except:
                pass
            # midpoint at the end of word_j
            step_count = 0
            try:
                while True:
                    
                    left_half = word[-step_count*2:-step_count]
                    right_half = get_reversed_substring(word_j, len(word)-step_count, len(word))
                    if step_count != 0 and left_half == right_half:
                        check_substrings(word[:len(word)-step_count*2], word_j, if_start=False)
                    left_half = word[-step_count*2-1:-step_count-1]
                    if left_half == right_half:
                        check_substrings(word[:-step_count*2-1], word_j, if_start=False)
                    step_count += 1
                    assert step_count <= len(word)//2
            except:
                pass
        return ret
```