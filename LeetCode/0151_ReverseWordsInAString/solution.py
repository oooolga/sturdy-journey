class Solution:
    def reverseWords(self, s: str) -> str:
        pointerA = pointerB = 0
        ret = ''
        while pointerA > -len(s) and pointerB > -len(s):
            if pointerB <= pointerA:
                if s[pointerB-1] == ' ':
                    pointerB -= 1
                else:
                    pointerA = pointerB - 1
            else:
                if s[pointerA-1] != ' ':
                    pointerA -= 1
                else:
                    if ret:
                        ret += ' ' + s[len(s)+pointerA:len(s)+pointerB]
                    else:
                        ret = s[len(s)+pointerA:len(s)+pointerB]
                    pointerB = pointerA-1
        if pointerA < pointerB:
            if ret:
                ret += ' ' + s[len(s)+pointerA:len(s)+pointerB]
            else:
                ret = s[len(s)+pointerA:len(s)+pointerB]
        return ret