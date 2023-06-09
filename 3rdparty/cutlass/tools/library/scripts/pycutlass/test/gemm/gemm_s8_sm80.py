import pycutlass
from pycutlass import *
from pycutlass.epilogue import LinearCombinationClamp
from pycutlass.test import *
import unittest

from pycutlass.test.gemm_testbed import test_all_gemm

class GemmS8TensorOpF32Sm80(unittest.TestCase):
    def test_SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32_64x64x64_32x32x64(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 32],
            element_a=cutlass.int8, element_b=cutlass.int8,
            element_accumulator=cutlass.int32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add_saturate
        )

        tile_description = TileDescription(
            threadblock_shape=[64, 64, 64],
            stages=6, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajorInterleaved32,
            alignment=16
        )
        B = TensorDescription(
            element=cutlass.int8, layout=cutlass.RowMajorInterleaved32,
            alignment=16
        )
        C = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajorInterleaved32,
            alignment=8
        )

        epilogue_functor = FastLinearCombinationClamp(
            C.element, C.alignment
        )
        
        swizzling_functor = cutlass.IdentitySwizzle1

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C,
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
        )

        self.assertTrue(test_all_gemm(operation, "interleaved"))
    
    def test_SM80_Device_Gemm_s8t_s8n_s8t_tensor_op_s32_256x128x128_64x64x128(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 32],
            element_a=cutlass.int8, element_b=cutlass.int8,
            element_accumulator=cutlass.int32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 128],
            stages=3, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.int8, layout=cutlass.RowMajor,
            alignment=16
        )
        B = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajor,
            alignment=16
        )
        C = TensorDescription(
            element=cutlass.int8, layout=cutlass.RowMajor,
            alignment=16
        )

        epilogue_functor = FastLinearCombinationClamp(
            C.element, C.alignment
        )
        
        swizzling_functor = cutlass.IdentitySwizzle1

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C,
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
        )

        self.assertTrue(test_all_gemm(operation, "multistage"))
    
    def test_SM80_Device_Gemm_s8t_s8n_s8n_tensor_op_s32_128x128x128_64x64x128(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 32],
            element_a=cutlass.int8, element_b=cutlass.int8,
            element_accumulator=cutlass.int32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 128],
            stages=3, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.int8, layout=cutlass.RowMajor,
            alignment=16
        )
        B = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajor,
            alignment=16
        )
        C = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajor,
            alignment=16
        )

        epilogue_functor = FastLinearCombinationClamp(
            C.element, C.alignment
        )
        
        swizzling_functor = cutlass.IdentitySwizzle1

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C,
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
        )

        self.assertTrue(test_all_gemm(operation, "multistage"))
    
    def test_SM80_Device_Gemm_s8t_s8n_s32n_tensor_op_s32_128x128x128_64x64x128(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 32],
            element_a=cutlass.int8, element_b=cutlass.int8,
            element_accumulator=cutlass.int32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 128],
            stages=3, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.int8, layout=cutlass.RowMajor,
            alignment=16
        )
        B = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajor,
            alignment=16
        )
        C = TensorDescription(
            element=cutlass.int32, layout=cutlass.ColumnMajor,
            alignment=4
        )

        element_epilogue = cutlass.int32

        epilogue_functor = LinearCombinationClamp(
            C.element, C.alignment, math_inst.element_accumulator, 
            element_epilogue
        )
        
        swizzling_functor = cutlass.IdentitySwizzle1

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C, 
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
        )

        self.assertTrue(test_all_gemm(operation, "multistage"))
    
    def test_SM80_Device_Gemm_s8t_s8n_s32t_tensor_op_s32_128x128x128_64x64x128(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 32],
            element_a=cutlass.int8, element_b=cutlass.int8,
            element_accumulator=cutlass.int32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 128],
            stages=3, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.int8, layout=cutlass.RowMajor,
            alignment=16
        )
        B = TensorDescription(
            element=cutlass.int8, layout=cutlass.ColumnMajor,
            alignment=16
        )
        C = TensorDescription(
            element=cutlass.int32, layout=cutlass.RowMajor,
            alignment=4
        )

        element_epilogue = cutlass.int32

        epilogue_functor = LinearCombinationClamp(
            C.element, C.alignment, math_inst.element_accumulator, 
            element_epilogue
        )
        
        swizzling_functor = cutlass.IdentitySwizzle1

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C,
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
        )

        self.assertTrue(test_all_gemm(operation, "multistage"))
    



if __name__ == '__main__':
    pycutlass.get_memory_pool(2**24, 2**24)
    unittest.main()
