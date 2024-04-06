class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates = sorted(candidates)
        
        def helper(candidates, target, candidate_i):
            candidate_val = candidates[candidate_i]
            remainders = {target-i:[candidate_val]*(i//candidate_val) for i in range(0, target+1, candidate_val)}

            ret = []
            for k in remainders:
                if k == 0:
                    ret.append(remainders[k])
                elif candidate_i > 0:
                    k_ret = helper(candidates, k, candidate_i-1)
                    for tmp in k_ret:
                        ret.append(tmp+remainders[k])
            return ret
        
        return helper(candidates, target, len(candidates)-1)

if __name__ == "__main__":
    testcase1 = {"candidates": [2,3,6,7],
                 "target": 7}
    sol = Solution()
    print(sol.combinationSum(**testcase1))