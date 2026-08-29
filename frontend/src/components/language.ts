/**
 * Best-effort language label for a code block.
 *
 * Its own module rather than living in CodeBlock.tsx so it stays testable:
 * `node --test` strips types from `.ts` but cannot parse JSX.
 *
 * Returns null rather than guessing. An answer written as pseudocode is common
 * in interviews, and labelling it "Python" is worse than leaving it unlabelled.
 */
const LANGUAGE_HINTS: [RegExp, string][] = [
  [/^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER)\b/im, "SQL"],
  [/^\s*(?:def|class)\s+\w+|^\s*from\s+\w+\s+import\b|^\s*import\s+\w+$/m, "Python"],
  [/^\s*(?:function|const|let|var)\s|=>\s*\{/m, "JavaScript"],
  [/\b(?:public|private|protected)\s+(?:static\s+)?(?:class|void|int|String)\b/m, "Java"],
  [/#include\s*<|std::|->\s*\w+\s*\{/m, "C++"],
];

export function detectLanguage(code: string): string | null {
  for (const [pattern, label] of LANGUAGE_HINTS) {
    if (pattern.test(code)) return label;
  }
  return null;
}
