# [Combination Sum](https://leetcode.com/problems/combination-sum/description/)

Given an array of **distinct** integers `candidates` and a target integer `target`, return a list of **all unique combinations** of `candidates` where the chosen numbers sum to `target`. You may return the combinations in any order.

The same number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.

## Solution (beats 93%)

*Run time: O(nlogn) where n represents the size of `candidates`. It's important to note that the number of unique combinations we can generate is capped at 150. Therefore, the runtime of the recursive function within our algorithm approximates to O(150n).*

To ensure that the combinations returned by our algorithm are unique, we start by sorting the `candidates`. 

Next, we repeatedly subtract the **largest** `candidate` from the `target` value until we can no longer subtract from `target` (`target` becomes non-positive). During each subtraction, we keep track of the remainder.

Then, for each computed remainder, we do the same thing again but this time using the **second biggest** `candidate` number, subtracting it repeatedly from the remainder value.

We carry on with this method for each candidate number. Whenever we manage to reduce the remainder to 0 by subtracting candidates, we append the sequence of candidate numbers that got us there to the `ret` list.

```
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
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
```