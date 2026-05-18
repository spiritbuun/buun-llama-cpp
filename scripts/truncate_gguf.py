#!/usr/bin/env python3
"""Create a truncated GGUF with only the first N layers.
Uses gguf-py's reader for parsing but writes raw bytes for reliability.
Usage: python truncate_gguf.py input.gguf output.gguf --n-layers 8
"""
import sys, struct, argparse, mmap
sys.path.insert(0, '/root/exp-boundary-v/gguf-py')
from gguf import GGUFReader
import numpy as np

GGUF_MAGIC = b'GGUF'

def write_string(f, s):
	b = s.encode('utf-8')
	f.write(struct.pack('<Q', len(b)))
	f.write(b)

def write_kv_string(f, key, val):
	write_string(f, key)
	f.write(struct.pack('<I', 8))  # STRING type
	write_string(f, val)

def write_kv_uint32(f, key, val):
	write_string(f, key)
	f.write(struct.pack('<I', 4))  # UINT32 type
	f.write(struct.pack('<I', val))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('input')
	parser.add_argument('output')
	parser.add_argument('--n-layers', type=int, required=True)
	args = parser.parse_args()

	reader = GGUFReader(args.input)

	# find architecture and block count
	arch_field = reader.fields.get('general.architecture')
	arch = str(bytes(arch_field.parts[-1]), 'utf-8')
	block_count_key = f'{arch}.block_count'
	bc_field = reader.fields[block_count_key]
	n_orig = struct.unpack('<I', bytes(bc_field.parts[-1])[:4])[0]
	n_keep = args.n_layers
	print(f"Architecture: {arch}, layers: {n_orig} -> {n_keep}")

	# identify tensors to keep
	keep_tensors = []
	for t in reader.tensors:
		if 'blk.' in t.name:
			idx = int(t.name.split('.')[1])
			if idx >= n_keep:
				continue
		keep_tensors.append(t)
	print(f"Keeping {len(keep_tensors)}/{len(reader.tensors)} tensors")

	with open(args.input, 'rb') as fin, open(args.output, 'wb') as fout:
		# Read header from input: magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24 bytes
		magic = fin.read(4)
		assert magic == GGUF_MAGIC, f"Bad magic: {magic}"
		version = struct.unpack('<I', fin.read(4))[0]
		n_tensors_orig = struct.unpack('<Q', fin.read(8))[0]
		n_kv = struct.unpack('<Q', fin.read(8))[0]
		print(f"GGUF v{version}, {n_tensors_orig} tensors, {n_kv} KV pairs")

		# Write output header
		fout.write(GGUF_MAGIC)
		fout.write(struct.pack('<I', version))
		fout.write(struct.pack('<Q', len(keep_tensors)))  # modified tensor count
		fout.write(struct.pack('<Q', n_kv))  # same KV count

		# Read and copy each KV pair, patching block_count

		# Read and copy each KV pair
		for i in range(n_kv):
			# Read key (string: len_u64 + bytes)
			key_len = struct.unpack('<Q', fin.read(8))[0]
			key_data = fin.read(key_len)
			key_str = key_data.decode('utf-8')

			# Read value type
			vtype = struct.unpack('<I', fin.read(4))[0]

			# Read value based on type
			if vtype == 8:  # STRING
				val_len = struct.unpack('<Q', fin.read(8))[0]
				val_data = fin.read(val_len)
				# write
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(struct.pack('<Q', val_len))
				fout.write(val_data)
			elif vtype == 4:  # UINT32
				val = struct.unpack('<I', fin.read(4))[0]
				if key_str == block_count_key:
					val = n_keep
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(struct.pack('<I', val))
			elif vtype == 5:  # INT32
				val_data = fin.read(4)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 6:  # FLOAT32
				val_data = fin.read(4)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 7:  # BOOL
				val_data = fin.read(1)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 10:  # UINT64
				val_data = fin.read(8)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 12:  # FLOAT64
				val_data = fin.read(8)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 9:  # ARRAY
				# array header: element_type(4) + count(8)
				arr_type = struct.unpack('<I', fin.read(4))[0]
				arr_count = struct.unpack('<Q', fin.read(8))[0]
				# write array header
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(struct.pack('<I', arr_type))
				fout.write(struct.pack('<Q', arr_count))
				# copy array elements
				if arr_type == 8:  # STRING array
					for _ in range(arr_count):
						slen = struct.unpack('<Q', fin.read(8))[0]
						sdata = fin.read(slen)
						fout.write(struct.pack('<Q', slen))
						fout.write(sdata)
				elif arr_type == 4:  # UINT32 array
					data = fin.read(4 * arr_count)
					fout.write(data)
				elif arr_type == 5:  # INT32 array
					data = fin.read(4 * arr_count)
					fout.write(data)
				elif arr_type == 6:  # FLOAT32 array
					data = fin.read(4 * arr_count)
					fout.write(data)
				elif arr_type == 10:  # UINT64 array
					data = fin.read(8 * arr_count)
					fout.write(data)
				else:
					print(f"  ERROR: unknown array element type {arr_type} in {key_str}")
					return 1
			elif vtype == 2:  # UINT8
				val_data = fin.read(1)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 3:  # INT8
				val_data = fin.read(1)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 0:  # UINT16
				val_data = fin.read(2)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			elif vtype == 1:  # INT16
				val_data = fin.read(2)
				fout.write(struct.pack('<Q', key_len))
				fout.write(key_data)
				fout.write(struct.pack('<I', vtype))
				fout.write(val_data)
			else:
				print(f"  ERROR: unknown value type {vtype} for key {key_str}")
				return 1

		print(f"KV section written, file pos: {fout.tell()}")

		# Now we need to figure out where tensor info starts in the input
		# tensor_info_start = current position in fin
		tensor_info_start_in = fin.tell()
		print(f"Tensor info starts at input offset: {tensor_info_start_in}")

		# Write tensor info for kept tensors
		# We need to compute new offsets for tensor data
		# First, figure out the tensor data alignment
		# After all tensor info entries, data is padded to ALIGNMENT (32 or 64 bytes)
		ALIGNMENT = 32  # GGUF v3 uses 32-byte alignment

		# Compute size of tensor info section for kept tensors
		# Each entry: name_len(8) + name + n_dims(4) + dims(n_dims*8) + type(4) + offset(8)
		tensor_info_size = 0
		for t in keep_tensors:
			name_bytes = t.name.encode('utf-8')
			n_dims = len(t.shape)
			tensor_info_size += 8 + len(name_bytes) + 4 + n_dims * 8 + 4 + 8

		# Data section starts at aligned position after header + kv + tensor_info
		data_section_file_offset = fout.tell() + tensor_info_size
		padding_needed = (ALIGNMENT - (data_section_file_offset % ALIGNMENT)) % ALIGNMENT
		data_section_file_offset += padding_needed

		# Write tensor info with new offsets
		current_data_offset = 0  # offset within data section
		tensor_data_offsets = []  # (input_data_ptr, size, output_offset)

		for t in keep_tensors:
			name_bytes = t.name.encode('utf-8')
			n_dims = len(t.shape)

			fout.write(struct.pack('<Q', len(name_bytes)))
			fout.write(name_bytes)
			fout.write(struct.pack('<I', n_dims))
			for d in t.shape:
				fout.write(struct.pack('<Q', int(d)))
			fout.write(struct.pack('<I', t.tensor_type))
			fout.write(struct.pack('<Q', current_data_offset))

			# align individual tensors to ALIGNMENT
			size = t.n_bytes
			tensor_data_offsets.append((t, current_data_offset))
			current_data_offset += size
			# align next tensor
			current_data_offset = ((current_data_offset + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

		# Write alignment padding
		fout.write(b'\x00' * padding_needed)

		print(f"Tensor info written, data section at offset: {fout.tell()}")
		assert fout.tell() == data_section_file_offset

		# Write tensor data
		for t, offset in tensor_data_offsets:
			# t.data is a numpy memmap view into the original file
			data = bytes(t.data)
			fout.write(data)
			# pad to alignment
			pad = (ALIGNMENT - (len(data) % ALIGNMENT)) % ALIGNMENT
			fout.write(b'\x00' * pad)

		print(f"Done: {args.output} ({fout.tell() / 1024**3:.2f} GB)")

if __name__ == '__main__':
	sys.exit(main() or 0)
