NNRT              := $(shell pwd)/nn_runtime/nnrt

NNAPI_DELEGATE    := $(shell pwd)/nn_runtime/nnapi_delegate

OVXLIB		  :=$(shell pwd)/ovxlib

$(NNAPI_DELEGATE) : $(NNRT)
$(NNRT)		  : $(OVXLIB)
MAIN_MODULE    := $(NNRT) $(NNAPI_DELEGATE) $(OVXLIB)

.PHONY: install $(MAIN_MODULE)
install: $(MAIN_MODULE)

$(MAIN_MODULE):
	@$(MAKE) -C $@ -f makefile.linux
	@$(MAKE) -C $@ -f makefile.linux install

.PHONY: clean
clean:
	@echo "remove *.so and *.o"
	@rm *.so
	@cd $(NNRT);rm *.o
	@cd $(NNAPI_DELEGATE);rm *.o	
