# Contributing to llm-from-scratch

## Branching strategy

- Gunakan branch `development` untuk pekerjaan fitur dan perbaikan jangka panjang.
- Buat branch fitur baru dari `development` untuk setiap perubahan besar, misalnya `feature/tokenizer-improvement`.
- Setelah perubahan siap, ajukan pull request ke `development`.
- Ketika `development` stabil, merge ke `main`.

## Testing

Jalankan tes sebelum membuat PR:

```bash
python -m pytest
```

## Workflow umum

1. Tarik perubahan terbaru dari `main`:
   ```bash
git checkout main
git pull origin main
```
2. Buat branch baru dari `development`:
   ```bash
git checkout development
git pull origin development
git checkout -b feature/nama-fitur
```
3. Kembangkan dan jalankan tes secara lokal.
4. Push branch ke remote:
   ```bash
git push -u origin feature/nama-fitur
```
5. Buka pull request dari branch fitur ke `development`.
