# [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Return `true` if there is *a cycle in the linked list*. Otherwise, return `false`.

## Solution 
*Runtime: O(n)*

*Space: O(1)*

[Floyd's cycle detection (the tortoise and hare algirhtm)](https://math.stackexchange.com/questions/913499/proof-of-floyd-cycle-chasing-tortoise-and-hare).
## Code
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        hare_pointer, tortoise_pointer = head, head
        try:
            while True:
                hare_pointer = hare_pointer.next.next
                tortoise_pointer = tortoise_pointer.next
                if hare_pointer == tortoise_pointer:
                    return True
        except Exception as e:
            print(e)
            return False
```