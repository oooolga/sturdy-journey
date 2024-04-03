# [Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/)

You are given a string s and an array of strings words. All the strings of words are of the same length.

A concatenated substring in s is a substring that contains all the strings of any permutation of words concatenated.

- For example, if words = ["ab","cd","ef"], then "abcdef", "abefcd", "cdabef", "cdefab", "efabcd", and "efcdab" are all concatenated strings. "acdbef" is not a concatenated substring because it is not the concatenation of any permutation of words.

Return the starting indices of all the concatenated substrings in s. You can return the answer in any order.

## Attempt 1

In our demonstration using the testcase s="barfoothefoobarman" with words=["foo","bar"], the solution is approached in three main steps:

**Step 1:Initialization of Dictionaries**

First, we iterate through the list words to establish two essential dictionaries. The first dictionary, named `word_id`, assigns a unique identifier to each word, creating a mapping of word to id. The second dictionary, `word_count`, maintains a tally of the occurrences of each word within words. For the given example, these dictionaries are initialized as `word_id = {"foo": 1, "bar": 2}` and `word_count = {1: 1, 2: 1}`, respectively.

**Step 2: Creation of Word ID Sequence**

As we traverse the string s, a list named `s_seq` is constructed, beginning at index `i = w_l`, where `w_l` is the length of a word from words. This list keeps track of the sequences of word IDs corresponding to the words found immediately before `s[:i+1]`. In the context of our example, `s_seq` is populated as follows: `[[2], [], [], [2, 1], [], [], [], [], [], [1], [], [], [1, 2], [], [], []]`. This representation allows us to map each substring of `s` to the respective IDs of words from `words` that precede it.

**Step 3: Checking for Permutations in `s_seq`**

Finally, we iterate over `s_seq` to identify if any of its lists represent a permutation of the word IDs. 

```
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        s_l = len(s)
        w_l = len(words[0])
        num_w = len(words)

        from collections import defaultdict
        word_count = defaultdict(int)
        word_id = defaultdict(int)
        id_counter = 1
        for word in words:
            if word_id[word] == 0:
                word_id[word] = id_counter
                id_counter += 1
            word_count[word_id[word]] += 1
        
        s_seq = [[] for i in range(s_l-w_l+1)]

        for i in range(s_l-w_l+1):
            buff = word_id[s[i:i+w_l]] if s[i:i+w_l] in word_id else 0
            if buff>0:
                if i >= w_l:
                    s_seq[i] = s_seq[i-w_l]+[buff]
                else:
                    s_seq[i].append(buff)

        def check_permutation(id_list):
            tmp = word_count.copy()
            for id in id_list:
                tmp[id] -= 1
            return not any(tmp.values())

        ret = []
        for i, id_list in enumerate(s_seq):
            if len(id_list) >= num_w:
                if check_permutation(id_list[-num_w:]):
                    ret.append(i-w_l*(num_w-1))
        
        return ret
```