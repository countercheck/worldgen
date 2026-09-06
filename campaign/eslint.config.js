import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: ['**/dist/**', '**/node_modules/**', '**/src/palette.ts'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // The engine is full of exhaustive switches over discriminated unions — event
      // kinds, order kinds, terrain grades. A missing case must be a compile error, not
      // a silent fallthrough, so the checker is told to enforce it.
      '@typescript-eslint/switch-exhaustiveness-check': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
);
