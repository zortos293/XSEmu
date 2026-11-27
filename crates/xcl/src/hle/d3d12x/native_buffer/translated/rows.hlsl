cbuffer Params : register(b0) {
    uint records, stride, actionOffset, actionWords;
    uint rowBytes, spillBytes, ops, outputStride;
    uint2 arenaGuest, arenaNative;
    uint firstRecord, hasCount, spreadAt, counted;
};
RWByteAddressBuffer recipe : register(u4);
RWByteAddressBuffer arguments : register(u0);
RWByteAddressBuffer countBuffer : register(u1);
RWByteAddressBuffer rows : register(u2);
RWByteAddressBuffer commands : register(u3);
RWByteAddressBuffer over : register(u5);

uint2 addressAt(uint2 base, uint bytes) {
    uint lo = base.x + bytes;
    return uint2(lo, base.y + (lo < base.x));
}

[RootSignature("RootConstants(num32BitConstants=16,b0), UAV(u4), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u5)")]
[numthreads(64,1,1)]
void BuildRows(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x + firstRecord;
    uint active = records;
    if (hasCount != 0u) active = min(active,countBuffer.Load(0));
    if (i >= active) return;
    uint src = i * stride, dst = i * rowBytes, outAt = i * outputStride;
    for (uint b = 0; b < 512u + spillBytes; b += 4u) rows.Store(dst+b,recipe.Load(b));
    rows.Store2(dst,addressAt(arenaGuest, dst+512u));
    for (uint k = 0; k < ops; ++k) {
        uint4 op = recipe.Load4(512u+spillBytes+k*16u);
        for (uint j=0; j<op.z; ++j) {
            uint value = i;
            if (op.w == 0u) value = arguments.Load(src+op.x+j*4u);
            rows.Store(dst+op.y+j*4u,value);
        }
    }
    commands.Store2(outAt,addressAt(arenaNative,dst));
    for (uint a=0; a<actionWords; ++a) commands.Store(outAt+8u+4u*a,arguments.Load(src+actionOffset+4u*a));
}

[RootSignature("RootConstants(num32BitConstants=16,b0), UAV(u4), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u5)")]
[numthreads(1,1,1)]
void LimitDispatch() {
    uint active = records;
    if (hasCount != 0u) active = min(active, countBuffer.Load(0));
    uint remaining = firstRecord;
    for (uint i = 0; i < active; ++i) {
        uint at = i * outputStride + 8u;
        uint3 asked = uint3(commands.Load(at), commands.Load(at + 4u), commands.Load(at + 8u));
        uint x = min(asked.x, 65535u);
        uint y = min(asked.y, 65535u);
        uint z = min(asked.z, 65535u);
        if (max(asked.x, max(asked.y, asked.z)) > 65535u) {
            bool spread = spreadAt != 0u && asked.z <= 65535u && (asked.y == 0u || asked.x <= 4294836225u / asked.y);
            if (spread) {
                uint xy = asked.x * asked.y;
                x = (xy == 0u || asked.z == 0u) ? 0u : 65535u;
                y = (xy + 65534u) / 65535u;
                z = asked.z;
                rows.Store2(i * rowBytes + spreadAt, asked.xy);
            }
            if (counted != 0u) {
                over.InterlockedAdd(0u, 1u);
                over.InterlockedMax(4u, asked.x);
                over.InterlockedMax(8u, asked.y);
                over.InterlockedMax(12u, asked.z);
                if (!spread) over.InterlockedAdd(16u, 1u);
            }
        }
        if (x != 0u && y != 0u && z != 0u) {
            uint xy = x * y;
            if (z > remaining / xy) { x = 0u; y = 0u; z = 0u; }
            else remaining -= xy * z;
        }
        commands.Store(at, x);
        commands.Store(at + 4u, y);
        commands.Store(at + 8u, z);
    }
}

[RootSignature("RootConstants(num32BitConstants=16,b0), UAV(u4), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u5)")]
[numthreads(1,1,1)]
void LimitDraw() {
    uint active = records;
    if (hasCount != 0u) active = min(active, countBuffer.Load(0));
    uint remaining = firstRecord;
    for (uint i = 0; i < active; ++i) {
        uint at = i * outputStride + 8u;
        uint c = commands.Load(at);
        uint inst = commands.Load(at + 4u);
        if (c != 0u && inst != 0u) {
            if (inst > remaining / c) commands.Store2(at, uint2(0u, 0u));
            else remaining -= c * inst;
        }
    }
}
