# [Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/description/)

Given an array of points where `points[i] = [x_i, y_i]` represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.

## Solution
Starting with a point `p_i`, compute the slope between `p_i` and every other point. Identify the line that contains `p_i` and hosts the greatest number of points by counting how many points share the same slope with `p_i`. And repeat the same step for other points.

## Code
```
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        from collections import defaultdict

        def slope_calculator(p1, p2):
            return (p2[1]-p1[1])/(p2[0]-p1[0]) if p2[0] != p1[0] else inf

        max_points = 1
        for p1_i in range(len(points)):
            line_count = defaultdict(int)
            for p2_i in range(p1_i+1, len(points)):
                m = slope_calculator(points[p1_i], points[p2_i])
                line_count[m] += 1
            for k in line_count:
                max_points = max(max_points, line_count[k]+1)

        return max_points 
```