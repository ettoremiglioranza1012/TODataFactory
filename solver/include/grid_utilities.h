#pragma once

#include "definitions.h"

void initializeGridContext(struct gridContext *gc, const int nl);
void initializeGridContextWithBC(struct gridContext *gc, const int nl,
                                 const char *bc_file_path);
void setFixedDof_halo(struct gridContext *gc, const int l);
void setFixedDof_halo_from_file(struct gridContext *gc, const int l,
                                const char *bc_file_path);
void freeGridContext(struct gridContext *gc, const int nl);

void allocateStateField(const struct gridContext gc, const int l, CTYPE **v);
void allocateStateField_MTYPE(const struct gridContext gc, const int l,
                              MTYPE **v);
void allocateStateField_STYPE(const struct gridContext gc, const int l,
                              STYPE **v);
