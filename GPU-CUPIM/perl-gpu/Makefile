app:
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --parallel
test:
	cd build/tests && ctest --output-on-failure
clean:
	rm -rf build/*
