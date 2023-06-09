makefilestr = '''
all: residuals.so.1.0.1 residuals.so

residuals.so.1.0.1: residuals.c residuals.h 
	gcc -Wall -c -fPIC residuals.c
	gcc -shared -Wl,-soname,residuals.so.1 -o residuals.so.1.0.1 residuals.o

residuals.so: residuals.so.1.0.1
	ln -s residuals.so.1.0.1 residuals.so
	
clean:
	rm -f residuals.so.1.0.1 residuals.so residuals.o *~
'''
with open('Makefile1', 'w') as writer:
    writer.write(makefilestr)