# SharedMath

# 1. Сконфигурировать (статическая сборка по умолчанию)
cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/SharedMath

# Или как shared library:
cmake -B build -DSharedMath_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/SharedMath

# 2. Собрать
cmake --build build -j$(nproc)

# 3. Прогнать тесты
cmake --build build --target test   # или: ctest --test-dir build

# 4. Установить
cmake --install build

# 5. Упаковать (ZIP / DEB)
cmake --build build --target package