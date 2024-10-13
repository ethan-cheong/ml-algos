init:
	pip install -r requirements.txt

test:
	python -m unittest

clean:
	rm -rf __pycache__