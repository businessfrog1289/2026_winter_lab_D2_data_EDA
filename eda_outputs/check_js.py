path = r'C:\Users\COM\AppData\Local\Temp\claude\c--Users-COM-Documents-GitHub-2026-winter-lab-D2-data-EDA\1b4cb359-53e2-4f7e-823b-40ed29d92a99\scratchpad\d2_eda_report.html'
html = open(path, encoding='utf-8').read()
script = html.split('<script>',1)[1].rsplit('</script>',1)[0]
start = script.index('const DATA = ')
end = script.index('\n', start)
code = script[:start] + script[end:]

i = 0
n = len(code)
line = 1
stack = []
BACKSLASH = chr(92)
while i < n:
    c = code[i]
    if c == chr(10):
        line += 1
        i += 1
        continue
    if c in ('"', "'", '`'):
        quote = c
        j = i+1
        while j < n and code[j] != quote:
            if code[j] == BACKSLASH:
                j += 2
                continue
            if code[j] == chr(10):
                line += 1
            j += 1
        i = j+1
        continue
    if c == '/' and i+1<n and code[i+1] == '/':
        j = code.index(chr(10), i)
        i = j
        continue
    if c == '/' and i+1<n and code[i+1] == '*':
        j = code.index('*/', i)
        i = j+2
        continue
    if c == '(':
        stack.append(('(', line))
    elif c == ')':
        if stack and stack[-1][0]=='(':
            stack.pop()
        else:
            print('unexpected ) at line', line)
    elif c == '[':
        stack.append(('[', line))
    elif c == ']':
        if stack and stack[-1][0]=='[':
            stack.pop()
        else:
            print('unexpected ] at line', line)
    elif c == '{':
        stack.append(('{', line))
    elif c == '}':
        if stack and stack[-1][0]=='{':
            stack.pop()
        else:
            print('unexpected } at line', line)
    i += 1

print('remaining stack (unclosed):', stack[:20])
