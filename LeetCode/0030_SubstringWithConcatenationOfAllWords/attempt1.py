class Solution:
    def findSubstring(self, s, words):
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

if __name__ == '__main__':
    case1_s = "barfoothefoobarman"
    case1_words = ["foo", "bar"]

    sol = Solution()
    print(sol.findSubstring(case1_s, case1_words))