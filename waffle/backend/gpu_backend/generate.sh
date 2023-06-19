#!/bin/bash
#
# Pre: create obj directories beforehand
#
# ------------------------------------------------------------------------------
# NOTES
# =====
#
# * Here-delimiter with quotes does not do parameter/command substitution
#   e.g.:
#
#    cat <<'END'
#    Generated on: $(date)
#    END
#
# ------------------------------------------------------------------------------
# NOTES
# =====
#
# - Consider using make commands as here: https://stackoverflow.com/a/32699423
#
################################################################################

# Make list of Library object files
CPP_FILES=$(find Lib -name '*.cpp')
OBJ_CPP=$(echo "$CPP_FILES" | sed "s/\\.cpp$/\\.o  \\\\/g")

C_FILES=$(find Lib -name '*.c')
OBJ_C=$(echo "$C_FILES" | sed "s/\\.c$/\\.o  \\\\/g")

OBJ_TMP=$(echo "$OBJ_CPP
$OBJ_C
")
OBJ=$(echo "$OBJ_TMP" | sed "s/^Lib\\//  /g")
#echo $OBJ


TEST_FILES=$(find Tests -name '*.cpp')
CPP_OBJ_TEST_TMP=$(echo "$TEST_FILES" | sed "s/\\.cpp$/\\.o  \\\\/g")

C_TEST_FILES=$(find Tests -name '*.c')
C_OBJ_TEST_TMP=$(echo "$C_TEST_FILES" | sed "s/\\.c$/\\.o  \\\\/g")

OBJ_TEST_TMP=$(echo "$CPP_OBJ_TEST_TMP
$C_OBJ_TEST_TMP
")
OBJ_TEST=$(echo "$OBJ_TEST_TMP" | sed "s/^/  /g")

OUTSIDE_KERNELS_SUPPORT=$(find ../kernels/Support -name '*.cpp')

# TODO remove final line with two spaces (need bash equivalent of 'chomp')
OUTSIDE_KERNELS_EXTRA=$(echo "$OUTSIDE_KERNELS_SUPPORT" |sed "s/\\.cpp$/\\.o\\\\/g" |sed "s/^/ /")
#OUTSIDE_KERNELS_EXTRA=$(echo "$OUTSIDE_KERNELS_SUPPORT" | sed 's/\.\.\///g' | sed 's/\.cpp$/.o/g' | sed 's/^/ /')


# Get list of executables
# NOTE: grepping on 'main(' is not fool-proof, of course.
OUTKER1=$(grep -rl "main(" ../kernels)
OUTKER2=$(echo "$OUTKER1" | sed "s/\\.cpp$/  \\\\/g")
OUTKERNELS=$(echo "$OUTKER2" | sed "s/^.*\//  /g")

mkdir -p obj

cat << END > sources.mk
#
# This file is generated!  Editing it directly is a bad idea.
#
# Generated on: $(date)
#
###############################################################################

# Library Object files - only used for LIB
OBJ := \\
$OBJ

KERNELS_OUTSIDE := \\
$OUTKERNELS

KERNELS_OUTSIDE_EXTRA :=\\
$OUTSIDE_KERNELS_EXTRA

# support files for tests
TESTS_FILES := \\
$OBJ_TEST

END
