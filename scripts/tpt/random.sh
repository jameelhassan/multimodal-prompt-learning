Uzairs, with alignment
CUDA_VISIBLE_DEVICES=3 sh scripts/tpt/maple_xd_uzairLR2.sh oxford_flowers 1
CUDA_VISIBLE_DEVICES=3 sh scripts/tpt/maple_xd_uzairLR3.sh oxford_flowers 1
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_uzairLR5e4.sh food101 3
CUDA_VISIBLE_DEVICES=0 sh scripts/tpt/maple_xd_uzairLR4e2.sh stanford_cars 3

Mine with alignment
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_align_lr5e4.sh food101 3
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_align_lr4e2.sh ucf101 3
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_align_lr4e4.sh fgvc_aircraft 3
CUDA_VISIBLE_DEVICES=3 sh scripts/tpt/maple_xd_align_lr2e4.sh sun397 1
caltech101

Uzairs without alignment
CUDA_VISIBLE_DEVICES=1 sh scripts/tpt/maple_xd_uzair_Noalign_lr5e4.sh oxford_pets 2

Mine without alignment
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_Noalign_lr5e4.sh sun397 3
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_Noalign_lr4e2.sh sun397 3

CUDA_VISIBLE_DEVICES=3 sh scripts/tpt/maple_xd_align_lr2e4.sh eurosat 1

Mine without tpt with alignment
CUDA_VISIBLE_DEVICES=2 sh scripts/tpt/maple_xd_NoTPT.sh imagenet_a 1

sh scripts/tpt/maple_xd_align_lay1.sh
sh scripts/tpt/maple_xd_align_lay2.sh
sh scripts/tpt/maple_xd_align_lay3.sh

CUDA_VISIBLE_DEVICES=3 sh scripts/tpt/maple.sh imagenet 1 P50

CUDA_VISIBLE_DEVICES=0 sh scripts/tpt/maple_xd.sh stanford_cars 3 P50