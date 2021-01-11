#ifndef CYCLES_RISCV_H_
#define CYCLES_RISCV_H_

unsigned long getCycles(void){
	unsigned long numberOfCyclesExecuted;
	asm volatile ("rdcycle %0" : "=r"(numberOfCyclesExecuted));
	return numberOfCyclesExecuted;
}

#endif