# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeKLists(self, lists):
        ret_root = ListNode()
        k = len(lists)
        if lists:
            lists = [l for l in lists if l is not None]

            def merge_2_lists(l1, l2):
                merged_list_root = merged_list = ListNode()
                while l1 and l2:
                    if l1.val == l2.val:
                        merged_list.next = ListNode(l1.val, ListNode(l1.val))
                        merged_list = merged_list.next.next
                        l1, l2 = l1.next, l2.next
                    elif l1.val < l2.val:
                        merged_list.next = ListNode(l1.val)
                        merged_list = merged_list.next
                        l1 = l1.next
                    else:
                        merged_list.next = ListNode(l2.val)
                        merged_list = merged_list.next
                        l2 = l2.next
                if l1:
                    merged_list.next = l1
                elif l2:
                    merged_list.next = l2
                return merged_list_root.next

            def merge_k_lists(lists):
                if len(lists) < 2:
                    return lists[0] if len(lists) == 1 else None
                k = len(lists)
                left_list = merge_k_lists(lists[:k//2])
                right_list = merge_k_lists(lists[k//2:])
                return merge_2_lists(left_list, right_list)
            ret_root.next = merge_k_lists(lists)
        return ret_root.next

if __name__ == "__main__":
    testcase1 = [ListNode(1, ListNode(4, ListNode(5))), 
                 ListNode(1, ListNode(3, ListNode(4))), 
                 ListNode(2, ListNode(6))]
    testcase2 = []
    testcase3 = [None]
    s = Solution()
    temp = s.mergeKLists(testcase1)
    while temp:
        print(temp.val, end=" ")
        temp = temp.next
    print()
    print(s.mergeKLists(testcase2))
    print(s.mergeKLists(testcase3))