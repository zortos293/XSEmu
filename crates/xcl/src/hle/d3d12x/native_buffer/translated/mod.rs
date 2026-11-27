use crate::hle::d3d12x::console_list::{inline_for, Layout, Param};
use windows::Win32::Graphics::Direct3D12::*;
mod dispatch;
#[cfg(test)]
mod draw_test;
pub(crate) use dispatch::execute;

pub(crate) struct Rows {
    pub(crate) template: Vec<u32>,
    pub(crate) stride: u32,
    pub(crate) action: D3D12_INDIRECT_ARGUMENT_TYPE,
    action_at: u32,
    action_words: u32,
    spill: u32,
    ops: u32,
    pub(crate) row_bytes: u32,
    reset: Vec<std::ops::Range<usize>>,
    /// dispatches past 65,535 groups on an axis: the byte offset of io[`IO_GROUP_SPREAD`] in a row
    /// when they are spread over x and y (0: capped), and the GPU address of the title pipeline's
    /// counters (0: none) – `bind::indirect_spread`
    pub(crate) spread: (u32, u64),
}

pub(crate) struct Generator {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    limit: ID3D12PipelineState,
    draw_limit: ID3D12PipelineState,
}

struct Bindings {
    params: [u32; 14],
    recipe: u64,
    buffers: [u64; 4],
    spread: (u32, u64),
}

pub(crate) struct Span<'a> {
    pub(crate) resource: &'a ID3D12Resource,
    pub(crate) offset: u64,
}

impl Span<'_> {
    unsafe fn va(&self, bytes: u64, uav: bool) -> Option<u64> {
        let desc = self.resource.GetDesc();
        let va = self.resource.GetGPUVirtualAddress();
        if va == 0
            || self.offset % 16 != 0
            || self.offset.checked_add(bytes)? > desc.Width
            || desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            || (uav
                && !desc
                    .Flags
                    .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        {
            return None;
        }
        va.checked_add(self.offset)
    }
}

impl Generator {
    pub(crate) unsafe fn new(dev: &ID3D12Device) -> Result<Self, String> {
        let bytes = crate::gpu::recomp::dxc::compile(
            include_str!("rows.hlsl"),
            "BuildRows",
            "cs_6_0",
            &[],
        )?;
        let root: ID3D12RootSignature = dev
            .CreateRootSignature(0, &bytes)
            .map_err(|e| e.to_string())?;
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let made = dev
            .CreateComputePipelineState(&desc)
            .map_err(|e| e.to_string());
        std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
        let pso = made?;
        let kernel = |entry: &str| -> Result<ID3D12PipelineState, String> {
            let code = crate::gpu::recomp::dxc::compile(include_str!("rows.hlsl"), entry, "cs_6_0", &[])?;
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            let made = dev.CreateComputePipelineState(&desc).map_err(|e| e.to_string());
            std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
            made
        };
        let (limit, draw_limit) = (kernel("LimitDispatch")?, kernel("LimitDraw")?);
        Ok(Self { root, pso, limit, draw_limit })
    }

    unsafe fn bindings(
        model: &Rows,
        n: u32,
        guest_arena: u64,
        recipe: Span<'_>,
        args: Span<'_>,
        count: Option<Span<'_>>,
        arena: Span<'_>,
        commands: Span<'_>,
    ) -> Option<Bindings> {
        let bytes = u64::from(n) * u64::from(model.row_bytes);
        let va = arena.va(bytes, true)?;
        let backing = crate::hle::d3d12x::real::uma_window_buffer(
            guest_arena as usize,
            usize::try_from(bytes).ok()?,
        )?;
        if (backing.0, backing.1)
            != (
                windows::core::Interface::as_raw(arena.resource) as usize,
                arena.offset,
            )
        {
            return None;
        }
        let params = model.constants(n, guest_arena, va, count.is_some())?;
        let src = args.va(u64::from(n) * u64::from(model.stride), true)?;
        let num = match count {
            Some(c) => c.va(4, true)?,
            None => src,
        };
        let recipe = recipe.va(model.template.len() as u64 * 4, true)?;
        let out = commands.va(u64::from(n) * u64::from(model.output_stride()), true)?;
        Some(Bindings { params, recipe, buffers: [src, num, va, out], spread: model.spread })
    }

    unsafe fn record(&self, list: &ID3D12GraphicsCommandList, bindings: Bindings, draws: bool) {
        let mut params = bindings.params;
        let n = params[0];
        list.SetComputeRootSignature(&self.root);
        super::live::set_pipeline(list, &self.pso);
        list.SetComputeRootUnorderedAccessView(1, bindings.recipe);
        for (i, b) in bindings.buffers.into_iter().enumerate() {
            list.SetComputeRootUnorderedAccessView(2 + i as u32, b);
        }
        let (spread_at, counters) = bindings.spread;
        list.SetComputeRootUnorderedAccessView(6, if counters != 0 { counters } else { bindings.buffers[3] });
        list.SetComputeRoot32BitConstants(0, 2, [spread_at, (counters != 0) as u32].as_ptr().cast(), 14);
        let barrier = || {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }])
        };
        barrier();
        for start in (0..n).step_by(65535usize * 64) {
            params[12] = start;
            list.SetComputeRoot32BitConstants(0, 14, params.as_ptr().cast(), 0);
            super::live::dispatch(list, (n - start).div_ceil(64).min(65535), 1, 1);
        }
        barrier();
        use crate::hle::d3d12x::indirect::{DRAW_BUDGET, GROUP_BUDGET};
        let (limit, budget) = if draws { (&self.draw_limit, DRAW_BUDGET) } else { (&self.limit, GROUP_BUDGET) };
        super::live::set_pipeline(list, limit);
        list.SetComputeRoot32BitConstant(0, budget, 12);
        super::live::dispatch(list, 1, 1, 1);
        barrier();
    }
}

impl Rows {
    pub(crate) fn prepare(
        stride: u32,
        args: &[[u32; 4]],
        layout: &Layout,
        gfx: bool,
        user_sgprs: Option<u32>,
        base_args: &[u8],
        base_io: &[u32; 128],
    ) -> Option<Self> {
        if base_args.len() != layout.size as usize || stride % 4 != 0 {
            return None;
        }
        let inline = if gfx {
            layout.inline_gfx
        } else {
            inline_for(layout, user_sgprs)
        };
        let first = if gfx { 8 } else { 4 };
        let spill = layout.size.checked_sub(inline)?;
        if inline % 4 != 0 || spill % 4 != 0 || first + inline / 4 > 32 {
            return None;
        }
        let mut template = base_io.to_vec();
        for (i, w) in base_args[..inline as usize].chunks_exact(4).enumerate() {
            template[first as usize + i] = u32::from_le_bytes(w.try_into().ok()?);
        }
        for w in base_args[inline as usize..].chunks_exact(4) {
            template.push(u32::from_le_bytes(w.try_into().ok()?));
        }
        let (&[action, ..], commands) = args.split_last()?;
        let action_words = match action {
            0 => 4,
            1 => 5,
            2 | 10 => 3,
            _ => return None,
        };
        if (gfx && action == 2) || (!gfx && action != 2) {
            return None;
        }
        let (mut at, mut ops) = (0u32, 0u32);
        let mut reset = Vec::new();
        for &[kind, param, dest, n] in commands {
            let (packed, words, ordinal) = match (kind, layout.params.get(param as usize)?) {
                (5, Param::Constants { at, count }) if dest.checked_add(n)? <= *count && n > 0 => {
                    (at.checked_add(dest.checked_mul(4)?)?, n, 0)
                }
                (11, Param::Constants { at, count }) if dest < *count => {
                    (at.checked_add(dest.checked_mul(4)?)?, 1, 1)
                }
                (6..=8, Param::Descriptor { at }) => (*at, 2, 0),
                _ => return None,
            };
            let end = packed.checked_add(words.checked_mul(4)?)?;
            if packed % 4 != 0 || end > layout.size || (packed < inline && end > inline) {
                return None;
            }
            let pos = if packed < inline {
                first * 4 + packed
            } else {
                512 + (packed - inline)
            };
            reset.push(packed as usize..end as usize);
            template.extend([at, pos, words, ordinal]);
            if ordinal == 0 {
                at = at.checked_add(words.checked_mul(4)?)?
            }
            ops += 1;
        }
        if at.checked_add(4 * action_words)? > stride {
            return None;
        }
        let row_bytes = 512u32.checked_add(spill.max(16))?.checked_add(255)? & !255u32;
        Some(Self {
            template,
            stride,
            action: D3D12_INDIRECT_ARGUMENT_TYPE(action as i32),
            action_at: at,
            action_words,
            spill,
            ops,
            row_bytes,
            reset,
            spread: (0, 0),
        })
    }

    pub(crate) fn reset_arguments(&self, args: &mut [u8]) {
        for range in &self.reset {
            args[range.clone()].fill(0);
        }
    }

    pub(crate) fn constants(
        &self,
        records: u32,
        guest: u64,
        native: u64,
        has_count: bool,
    ) -> Option<[u32; 14]> {
        if guest == 0 || native == 0 || guest & 255 != 0 || native & 255 != 0 {
            return None;
        }
        let rows = u64::from(records.checked_mul(self.row_bytes)?);
        if guest.checked_add(rows)? > 1 << 48 {
            return None;
        }
        native.checked_add(rows)?;
        records.checked_mul(self.stride)?;
        records.checked_mul(self.output_stride())?;
        Some([
            records,
            self.stride,
            self.action_at,
            self.action_words,
            self.row_bytes,
            self.spill,
            self.ops,
            self.output_stride(),
            guest as u32,
            (guest >> 32) as u32,
            native as u32,
            (native >> 32) as u32,
            0,
            has_count as u32,
        ])
    }

    pub(crate) fn output_stride(&self) -> u32 {
        8 + 4 * self.action_words
    }

    pub(crate) unsafe fn signature(
        &self,
        dev: &ID3D12Device,
        abi: &ID3D12RootSignature,
    ) -> windows::core::Result<ID3D12CommandSignature> {
        let args = [
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    UnorderedAccessView: D3D12_INDIRECT_ARGUMENT_DESC_0_4 {
                        RootParameterIndex: 0,
                    },
                },
            },
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: self.action,
                ..Default::default()
            },
        ];
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: self.output_stride(),
            NumArgumentDescs: 2,
            pArgumentDescs: args.as_ptr(),
            NodeMask: 0,
        };
        let mut signature = None;
        dev.CreateCommandSignature(&desc, abi, &mut signature)?;
        signature.ok_or_else(|| {
            windows::core::Error::from_hresult(windows::core::HRESULT(0x80004005u32 as i32))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_uav_guest_words_go_inline_or_to_spill_without_host_va_translation() {
        let layout = Layout {
            params: vec![
                Param::Descriptor { at: 0 },
                Param::Constants { at: 8, count: 1 },
                Param::Descriptor { at: 12 },
            ],
            size: 20,
            inline: 20,
            inline_gfx: 20,
            ..Default::default()
        };
        let guest = [0x1234u32, 3, 77, 0x4321, 5]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut io = [0u32; 128];
        io[118] = 39;
        let sig = [[8, 0, 0, 0], [8, 2, 0, 0], [11, 1, 0, 0], [2, 0, 0, 0]];
        let model = Rows::prepare(28, &sig, &layout, false, Some(7), &guest, &io).unwrap();
        assert_eq!(model.template[118], 39);
        assert_eq!(&model.template[4..7], &[0x1234, 3, 77]);
        assert_eq!(&model.template[128..130], &[0x4321, 5]);
        assert_eq!(
            &model.template[130..],
            &[0, 16, 2, 0, 8, 512, 2, 0, 16, 24, 1, 1]
        );
        assert_eq!((model.action_at, model.output_stride()), (16, 20));
        let mut packed = (1u8..=20).collect::<Vec<_>>();
        model.reset_arguments(&mut packed);
        assert_eq!(&packed[..], &[0u8; 20][..]);
        let plain = Rows::prepare(12, &[[2, 0, 0, 0]], &layout, false, Some(7), &guest, &io).unwrap();
        let mut kept = guest.clone();
        plain.reset_arguments(&mut kept);
        assert_eq!((kept, plain.action_at, plain.output_stride(), plain.template.len()), (guest.clone(), 0, 20, 130));
        assert!(Rows::prepare(28, &sig, &layout, true, None, &guest, &io).is_none());
        assert!(Rows::prepare(16, &[[9, 0, 0, 0]], &layout, false, None, &guest, &io).is_none());
    }
}
