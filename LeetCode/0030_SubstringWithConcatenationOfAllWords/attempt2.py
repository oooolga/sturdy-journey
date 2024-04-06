class Solution:
    def findSubstring(self, s: str, words: list[str]) -> list[int]:
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

if __name__ == '__main__':
    case1_s = "barfoothefoobarman"
    case1_words = ["foo", "bar"]

    sol = Solution()
    print(sol.findSubstring(case1_s, case1_words))