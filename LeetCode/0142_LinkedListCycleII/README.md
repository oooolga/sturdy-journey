# [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

Given the `head` of a linked list, return the node where the cycle begins. If there is no cycle, return `null`.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to (0-indexed). It is `-1` if there is no cycle. Note that `pos` is not passed as a parameter.

Do not modify the linked list.

## Solution
Implement the Tortoise and Hare algorithm, returning None if the `hare_pointer` and `tortoise_pointer` do not meet. Continue advancing the tortoise_pointer, and initiate a second tortoise pointer, `tortoise_pointer2`, from the head. The cycle's starting node is identified at the point where `tortoise_pointer` and `tortoise_pointer2` meet.

## Proof
Define `node_m` as the meeting point of the `hare_pointer` and `tortoise_pointer`. Denote `node_s` as the starting node of the cycle. Let $n$ represent the cycle's length, $m$ the distance from `node_s` to `node_m`, and $e$ the distance from the start of the link list to `node_s`.

When the `hare_pointer` and `tortoise_pointer` intersect, the distance covered by the `hare_pointer` is $n \times x + m + e$, with $x$ denoting the number of laps the `hare_pointer` completes around the cycle before encountering the `tortoise_pointer`. Meanwhile, the `tortoise_pointer` traverses a distance of $m + e$.

Given that the `hare_pointer` moves at twice the speed of the `tortoise_pointer`, the following equations are established as 

$$
\begin{split}
  n \times x + m + e &= 2(m+e)\\
  n \times x &= m+e \\
  n \times (x-1) + (n-m) + m &= m+e\\
  n \times (x-1) + (n-m) &= e
\end{split}
$$

Therefore, when `tortoise_pointer2` travels a distance of $e$, it reaches the start of the linked list. At this point, `tortoise_pointer` would have traveled a distance of $n - m$ to the cycle's start and completed $x - 1$ laps around the cycle. Thus, after $e$ steps, both pointers would meet at the start of the cycle.

## Code
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        hare, tortoise = head, head
        try:
            while True:
                hare = hare.next.next
                tortoise = tortoise.next
                if hare == tortoise:
                    tortoise2 = head
                    while tortoise2 != tortoise:
                        tortoise = tortoise.next
                        tortoise2 = tortoise2.next
                    return tortoise
        except:
            return None
```