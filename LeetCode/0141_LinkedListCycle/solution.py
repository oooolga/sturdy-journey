# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head):
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

if __name__ == "__main__":
    sol = Solution()
    head = ListNode(3)
    head.next = ListNode(2)
    head.next.next = ListNode(0)
    head.next.next.next = ListNode(-4)
    head.next.next.next.next = head.next
    print(sol.hasCycle(head))  # True