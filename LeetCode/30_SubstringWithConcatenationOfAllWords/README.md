# [Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/)

You are given a string s and an array of strings words. All the strings of words are of the same length.

A concatenated substring in s is a substring that contains all the strings of any permutation of words concatenated.

- For example, if words = ["ab","cd","ef"], then "abcdef", "abefcd", "cdabef", "cdefab", "efabcd", and "efcdab" are all concatenated strings. "acdbef" is not a concatenated substring because it is not the concatenation of any permutation of words.

Return the starting indices of all the concatenated substrings in s. You can return the answer in any order.

## Attempt 2
*Runtime: O(len(s)\*len(words))*

*Space: O(len(words)+len(words[0]))*

In our demonstration using the testcase s="barfoothefoobarman" with words=["foo","bar"], the solution is approached in three main steps:

**Step 1:Initialization of Dictionaries**

First, we iterate through the list words to establish two essential dictionaries. The first dictionary, named `word_id`, assigns a unique identifier to each word, creating a mapping of word to id. The second dictionary, `word_count`, maintains a tally of the occurrences of each word within words. For the given example, these dictionaries are initialized as `word_id = {"foo": 1, "bar": 2}` and `word_count = {1: 1, 2: 1}`, respectively.

**Step 2: Iterate Over `s` and Find Substring with Concatenation of All Words**

*Step 2a*: As we navigate through the string `s`, we employ a queue named `prevs` which is designed to contain precisely `w_l` elements, with `w_l` being the length of the words in our list. This queue, prevs, plays a critical role in mapping out the occurrence and sequence of words within specific substring of `s`. These substrings are defined in a rolling window manner relative to our current index `i` in `s`, ranging from `s[i-w_l*n_w:i-w_l]` and extending up to `s[i-w_l*(n_w-1)+1:i+1]`, where `n_w` denotes the number of words in `words`. To be specific, the data held in each position of `prevs` encapsulates a list of words discovered at the terminal portion of its corresponding segment, preserving the order of their discovery. 

Each time we move to a new position `i` in `s`, we check the segment ending at `i`. We dequeue the oldest list from `prevs` (associated with the segment `s[i-w_l*n_w:i-w_l]`) and see if the new segment `s[i:i+w_l]` matches any of our words. If it does, we may add this word to the list we just removed and put it back at the end of `prevs` (more details related to this step will be explained in *Step 2b* ). If not, we add an empty list to the end of `prevs`. This way, `prevs` keeps an updated record of which words were found and in what order, focusing on the ends of the segments we're examining.

*Step 2b*: 
When adding a new word to a previously formed list within our process, we perform two critical checks: 1. the length of the list and 2. valid permutation substring.

1. For the list length check, when we add a new word to the list, we make sure the list doesn't get longer than `n_w`, which is the total number of words we're looking for. If adding the new word makes the list too long, we remove the first word from the list before adding the new one. This way, the list always has `n_w` words or fewer, helping us keep track of word sequences properly.
2. For the valid permutation substring check, this step is initiated when the updated list reaches a length of `n_w`. If we had to remove an old word to add a new one, we first see if the old and new words are the same. If they are, then we've found a correct sequence. If not, we have to manually check if the new list still forms a correct sequence of words.
```
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        s_l = len(s)
        w_l = len(words[0])
        num_w = len(words)

        # Step 1
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
        ret = set()
        
        # Step 2a
        def add_new_id(prev_seq, new_id, if_prev_permute):
            if len(prev_seq) < num_w-1:
                return False, prev_seq+[new_id]
            elif len(prev_seq) == num_w-1:
                new_list = prev_seq+[new_id]
            else:
                new_list = prev_seq[1:]+[new_id]
            if not if_prev_permute:
                return check_permutation(new_list), new_list
            else:
                return prev_seq[0]==new_id, new_list
        # Step 2b
        def check_permutation(id_list):
            tmp = word_count.copy()
            for i, id in enumerate(id_list):
                tmp[id] -= 1
                if tmp[id] < 0:
                    return False
            return True
        
        prevs = [[]]*w_l
        # Step 2
        for i in range(s_l-w_l+1):
            new_id = word_id[s[i:i+w_l]] if s[i:i+w_l] in word_id else 0
            read_write_pointer = i % w_l
            if new_id>0:
                if_permute, prevs[read_write_pointer] = add_new_id(prevs[read_write_pointer],
                                                                   new_id,
                                                                   bool(i-w_l*num_w in ret))
                if if_permute:
                    ret.add(i-(num_w-1)*w_l)
            else:
                prevs[read_write_pointer] = []
        return list(ret)

```
## Attempt 1

**Step 1: Same as Attempt 2**

**Step 2: Creation of Word ID Sequence**

As we traverse the string `s`, a list named `s_seq` is constructed, beginning at index `i = w_l`, where `w_l` is the length of a word from words. This list keeps track of the sequences of word IDs corresponding to the words found immediately before `s[:i+1]`. In the context of our example, `s_seq` is populated as follows: `[[2], [], [], [2, 1], [], [], [], [], [], [1], [], [], [1, 2], [], [], []]`. This representation allows us to map each substring of `s` to the respective IDs of words from `words` that precede it.

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