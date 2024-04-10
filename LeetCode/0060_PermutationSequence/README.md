# [Permutation Sequence](https://leetcode.com/problems/permutation-sequence/description/)

The set `[1, 2, 3, ..., n]` contains a total of `n!` unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for `n = 3`:

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`
   
Given `n` and `k`, return the `k`th permutation sequence.

## Solution

Beginning with the first element in an example where `n=3`, it's evident that the list of permutations can be segmented into three distinct chunks. Within each chunk, the value of the first element is consistently fixed at `1`, `2`, and `3`, respectively. This pattern becomes more obvious with the example of `n=4`:
1. `"1234"`
2. `"1243"`
3. `"1324"`
4. `"1342"`
5. `"1423"`
6. `"1432"`
------------
7. `"2134"`
8. `"2143"`
9.  `"2314"`
10. `"2341"`
11. `"2413"`
12. `"2431"`
------------
13. `"3124"`
14. `"3142"`
15. `"3214"`
16. `"3241"`
17. `"3412"`
18. `"3421"`
------------
19. `"4123"`
20. `"4132"`
21. `"4213"`
22. `"4231"`
23. `"4312"`
24. `"4321"`
    
In the case of `n=4`, dividing the permutation list based on the initial element results in four separate lists. The length of each list is `3!`, reflecting the number of permutations possible for the remaining three elements.

For the first chunk, where the initial element is `1`, further division based on the second element creates three distinct lists. Each of these lists contains permutations of the remaining two elements, resulting in a length of `2!` for each list.

1. `"1234"`
2. `"1243"`
------------
3. `"1324"`
4. `"1342"`
------------
5. `"1423"`
6. `"1432"`

In our method, we begin by establishing an `ordered_elements` list, which catalogs the elements available for selection in ascending order.  We build the final sequence, `ret`, one element at a time. To decide which element to pick for the `i`th spot, we divide `k` by the factorial of `(n-i)` (using `k//(n-i)!`). This calculation guides us in identifying which chunk we are currently in within the permutation list. The resulting chunk index corresponds to the position in the `ordered_elements` list from which we select our `i`th element.

Next, we adjust `k` by setting it to the remainder of `k` divided by `(n-i)!` (using `k % (n-i)!`), and remove the chosen element from the `ordered_elements` list.

## Code
```
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        factors, ret, ordered_elements = [1], "", []
        for i in range(1, n+1):
            factors.append(factors[-1]*i)
            ordered_elements.append(str(i))
        factors.pop()
        k = k-1
        for i in range(1, n+1):
            factor = factors.pop()
            element_idx, k = divmod(k, factor)
            ret += ordered_elements[element_idx]
            ordered_elements = ordered_elements[:element_idx] + ordered_elements[element_idx+1:]
        return ret
```