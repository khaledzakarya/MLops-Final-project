.PHONY: install-core install-cpu install-gpu quick-test

# 1. تثبيت الأساسيات فقط
install-core:
	pip install . --no-cache-dir

# 2. تثبيت بيئة الاختبار الخفيفة (CPU) للبيئات السحابية
install-cpu: install-core
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 3. تثبيت بيئة الإنتاج والتدريب (GPU)
install-gpu:
	pip install .[gpu] --no-cache-dir

# 4. تثبيت البيئة كاملة مع دعم جوجل درايف
install-full:
	pip install .[full] --no-cache-dir

# 5. أمر التيست السريع (يسحب النماذج ويختبر الكود مباشرة)
quick-test: install-cpu
	bash pull_models.sh
	python test_logits.py
