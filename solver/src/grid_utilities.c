#include "grid_utilities.h"

#include "local_matrix.h"

void setFixedDof_halo(struct gridContext *gc, const int l) {

  const int ncell = pow(2, l);
  const int32_t nelyc = (*gc).nely / ncell;
  const int32_t nelzc = (*gc).nelz / ncell;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  const int nzc = (nelzc + 1);
  const int nyc = (nelyc + 1);

  // classic cantilever
  // (*gc).fixedDofs[l].n = 3 * nyc * nzc;
  // (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) *
  // (*gc).fixedDofs[l].n); int offset = 0; for (uint_fast32_t k = 1; k < (nzc +
  // 1); k++)
  //   for (uint_fast32_t j = 1; j < (nyc + 1); j++) {
  //     (*gc).fixedDofs[l].idx[offset + 0] =
  //         3 * (wrapyc * wrapzc + wrapyc * k + j) + 0;
  //     (*gc).fixedDofs[l].idx[offset + 1] =
  //         3 * (wrapyc * wrapzc + wrapyc * k + j) + 1;
  //     (*gc).fixedDofs[l].idx[offset + 2] =
  //         3 * (wrapyc * wrapzc + wrapyc * k + j) + 2;
  //     offset += 3;
  //   }

  // new cantilever
  const int nodelimit = (nelyc / 4) + 1;
  (*gc).fixedDofs[l].n = 3 * nzc * 2 * nodelimit;
  (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) * (*gc).fixedDofs[l].n);
  int offset = 0;
  const int i = 1;
  for (uint_fast32_t k = 1; k < (nzc + 1); k++) {
    for (uint_fast32_t j = 1; j < nodelimit + 1; j++) {
      (*gc).fixedDofs[l].idx[offset + 0] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 0;
      (*gc).fixedDofs[l].idx[offset + 1] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 1;
      (*gc).fixedDofs[l].idx[offset + 2] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 2;
      offset += 3;
    }
    for (uint_fast32_t j = (nyc + 1) - nodelimit; j < (nyc + 1); j++) {
      (*gc).fixedDofs[l].idx[offset + 0] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 0;
      (*gc).fixedDofs[l].idx[offset + 1] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 1;
      (*gc).fixedDofs[l].idx[offset + 2] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 2;
      offset += 3;
    }
  }
}

void setFixedDof_halo_from_file(struct gridContext *gc, const int l,
                                 const char *bc_file_path) {
  // ========================================
  // Read fixed DOF indices from binary file
  // ========================================
  // File format: int32 array of DOF indices
  // These are for level 0 (fine grid), need to scale for coarser levels

  if (bc_file_path == NULL) {
    // Fall back to hardcoded cantilever
    setFixedDof_halo(gc, l);
    return;
  }

  FILE *bc_file = fopen(bc_file_path, "rb");
  if (bc_file == NULL) {
    fprintf(stderr,
            "⚠️  Warning: Could not open BC file: %s, using fallback\n",
            bc_file_path);
    setFixedDof_halo(gc, l);
    return;
  }

  // Get file size to determine number of DOFs
  fseek(bc_file, 0, SEEK_END);
  long file_size = ftell(bc_file);
  fseek(bc_file, 0, SEEK_SET);

  int32_t num_dofs = file_size / sizeof(int32_t);
  if (num_dofs <= 0) {
    fprintf(stderr, "⚠️  Warning: Empty BC file, using fallback\n");
    fclose(bc_file);
    setFixedDof_halo(gc, l);
    return;
  }

  // Read DOF indices for level 0
  int32_t *fine_dofs = malloc(sizeof(int32_t) * num_dofs);
  size_t items_read = fread(fine_dofs, sizeof(int32_t), num_dofs, bc_file);
  fclose(bc_file);

  if (items_read != (size_t)num_dofs) {
    fprintf(stderr, "⚠️  Warning: Failed to read BC file, using fallback\n");
    free(fine_dofs);
    setFixedDof_halo(gc, l);
    return;
  }

  // For coarser levels, we need to map DOFs from fine to coarse
  const int ncell = pow(2, l);

  if (l == 0) {
    // Fine level - use DOFs directly
    (*gc).fixedDofs[l].n = num_dofs;
    (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) * num_dofs);

    for (int i = 0; i < num_dofs; i++) {
      (*gc).fixedDofs[l].idx[i] = (uint_fast32_t)fine_dofs[i];
    }

    printf("📍 BC Level %d: Read %d fixed DOFs from file\n", l, num_dofs);
  } else {
    // Coarser levels - need to map node indices
    // DOF = 3 * node_idx + component
    // node_idx = i * wrapy * wrapz + wrapy * k + j

    const int32_t nelyc = (*gc).nely / ncell;
    const int32_t nelzc = (*gc).nelz / ncell;

    const int paddingyc =
        (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
    const int paddingzc =
        (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

    const int wrapyc = nelyc + paddingyc + 3;
    const int wrapzc = nelzc + paddingzc + 3;

    // Fine grid wrap dimensions
    const int wrapy_fine = (*gc).wrapy;
    const int wrapz_fine = (*gc).wrapz;

    // Create a set of unique coarse DOFs
    // Use a temporary larger buffer and count unique entries
    uint_fast32_t *temp_dofs = malloc(sizeof(uint_fast32_t) * num_dofs);
    int coarse_count = 0;

    for (int d = 0; d < num_dofs; d++) {
      int32_t fine_dof = fine_dofs[d];
      int component = fine_dof % 3;
      int fine_node = fine_dof / 3;

      // Extract fine node coordinates
      int i_fine = fine_node / (wrapy_fine * wrapz_fine);
      int rem = fine_node % (wrapy_fine * wrapz_fine);
      int k_fine = rem / wrapy_fine;
      int j_fine = rem % wrapy_fine;

      // Map to coarse coordinates
      int i_coarse = (i_fine - 1) / ncell + 1;
      int j_coarse = (j_fine - 1) / ncell + 1;
      int k_coarse = (k_fine - 1) / ncell + 1;

      // Ensure within bounds
      if (i_coarse >= 1 && j_coarse >= 1 && k_coarse >= 1) {
        uint_fast32_t coarse_node =
            i_coarse * wrapyc * wrapzc + wrapyc * k_coarse + j_coarse;
        uint_fast32_t coarse_dof = 3 * coarse_node + component;

        // Check if already added (simple linear search for small counts)
        int found = 0;
        for (int c = 0; c < coarse_count; c++) {
          if (temp_dofs[c] == coarse_dof) {
            found = 1;
            break;
          }
        }

        if (!found) {
          temp_dofs[coarse_count++] = coarse_dof;
        }
      }
    }

    (*gc).fixedDofs[l].n = coarse_count;
    (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) * coarse_count);
    for (int i = 0; i < coarse_count; i++) {
      (*gc).fixedDofs[l].idx[i] = temp_dofs[i];
    }

    free(temp_dofs);
    printf("📍 BC Level %d: Mapped to %d fixed DOFs\n", l, coarse_count);
  }

  free(fine_dofs);
}

void initializeGridContext(struct gridContext *gc, const int nl) {
  // Initialize with NULL bc_file_path (uses hardcoded cantilever)
  initializeGridContextWithBC(gc, nl, NULL);
}

void initializeGridContextWithBC(struct gridContext *gc, const int nl,
                                  const char *bc_file_path) {

  const int paddingx =
      (STENCIL_SIZE_X - (((*gc).nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - (((*gc).nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - (((*gc).nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  (*gc).wrapx = (*gc).nelx + paddingx + 3;
  (*gc).wrapy = (*gc).nely + paddingy + 3;
  (*gc).wrapz = (*gc).nelz + paddingz + 3;

  (*gc).precomputedKE = malloc(sizeof(MTYPE *) * nl);
  (*gc).fixedDofs = malloc(sizeof(struct FixedDofs) * nl);

  for (int l = 0; l < nl; l++) {
    const int ncell = pow(2, l);
    const int pKESize = 24 * 24 * ncell * ncell * ncell;
    (*gc).precomputedKE[l] = malloc(sizeof(MTYPE) * pKESize);
    getKEsubspace((*gc).precomputedKE[l], (*gc).nu, l);

    // Use file-based BCs if path provided, otherwise hardcoded
    setFixedDof_halo_from_file(gc, l, bc_file_path);
  }
}

void freeGridContext(struct gridContext *gc, const int nl) {

  for (int l = 0; l < nl; l++) {
    free((*gc).precomputedKE[l]);
    free((*gc).fixedDofs[l].idx);
  }

  free((*gc).precomputedKE);
  free((*gc).fixedDofs);
}

void allocateStateField(const struct gridContext gc, const int l, CTYPE **v) {

  const int ncell = pow(2, l);

  const int nelx = gc.nelx / ncell;
  const int nely = gc.nely / ncell;
  const int nelz = gc.nelz / ncell;

  const int paddingx =
      (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapx = nelx + paddingx + 3;
  const int wrapy = nely + paddingy + 3;
  const int wrapz = nelz + paddingz + 3;
  const int ndof = 3 * wrapx * wrapy * wrapz;

  (*v) = malloc(sizeof(CTYPE) * ndof);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}

void allocateStateField_MTYPE(const struct gridContext gc, const int l,
                              MTYPE **v) {

  const int ncell = pow(2, l);

  const int nelx = gc.nelx / ncell;
  const int nely = gc.nely / ncell;
  const int nelz = gc.nelz / ncell;

  const int paddingx =
      (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapx = nelx + paddingx + 3;
  const int wrapy = nely + paddingy + 3;
  const int wrapz = nelz + paddingz + 3;
  const int ndof = 3 * wrapx * wrapy * wrapz;

  (*v) = malloc(sizeof(MTYPE) * ndof);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}

void allocateStateField_STYPE(const struct gridContext gc, const int l,
                              STYPE **v) {

  const int ncell = pow(2, l);

  const int nelx = gc.nelx / ncell;
  const int nely = gc.nely / ncell;
  const int nelz = gc.nelz / ncell;

  const int paddingx =
      (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapx = nelx + paddingx + 3;
  const int wrapy = nely + paddingy + 3;
  const int wrapz = nelz + paddingz + 3;
  const int ndof = 3 * wrapx * wrapy * wrapz;

  (*v) = malloc(sizeof(STYPE) * ndof);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}
