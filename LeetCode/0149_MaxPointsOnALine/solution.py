class Solution:
    def maxPoints(self, points: list[list[int]]) -> int:
        from collections import defaultdict

        def slope_calculator(p1, p2):
            return (p2[1]-p1[1])/(p2[0]-p1[0]) if p2[0] != p1[0] else float('inf')

        max_points = 1
        for p1_i in range(len(points)):
            line_count = defaultdict(int)
            for p2_i in range(p1_i+1, len(points)):
                m = slope_calculator(points[p1_i], points[p2_i])
                line_count[m] += 1
            for k in line_count:
                max_points = max(max_points, line_count[k]+1)

        return max_points 

if __name__ == '__main__':
    sol = Solution()
    testcase1 = [[1,1],[2,2],[3,3]]
    print(sol.maxPoints(testcase1))
    testcase2 = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    print(sol.maxPoints(testcase2))
    testcase3 = [[0,0],[0,0]]
    print(sol.maxPoints(testcase3))