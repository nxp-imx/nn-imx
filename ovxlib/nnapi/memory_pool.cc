#include "memory_pool.h"

#include <cstring>
#include <algorithm>

#include "vsi_nn_log.h"
#include "file_map_memory.h"

namespace mem_pool{

namespace alloc{
    /**
     * @brief Memory mapped block "allocation" information
     *
     */
    struct Mmp_block_infor {
        const ovxlib::Memory* address_; //!< which Memory Mapped Instance refered by this allocation request
        size_t offset_;                 //!< Internal offset of mapped space
        size_t len_;                    //!< How many bytes request
    };

    /**
     * @brief Normal block "allocation" information
     * Normal is relateive concept for small block, @see Sml_block_infor
     */
    struct Nml_block_infor {
        const void* address_;   //!< source memory address referenced by allocation request
        size_t len_;            //!< source memory length in bytes
    };

    /**
     * @brief Small block "allocation" information
     * When source memory length less than ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES
     * It should be treated as small block - copy source memory to internal managed memory
     */
    struct Sml_block_infor {
        const void* address_;
        size_t len_;
    };
}

bool operator==(const Ref& left, const Ref& right) {
    return (left.address_ == right.address_)
        && (left.len_ == right.len_)
        && (left.mem_type_ == right.mem_type_)
        && (left.index_ == right.index_)
        && (left.offset_ == right.offset_);
}

bool operator!=(const Ref& left, const Ref& right) {
    return !(left == right);
}

struct Ref_deleter{
    void operator()(Ref* memRef) const;
};

static const Ref INVALID_REF = {nullptr, 0, Type::INVALID, 0, 0};

template <typename T>
struct mem_type_of {
    static const int value = static_cast<int>(Type::CNT);
};

template <>
struct mem_type_of<alloc::Mmp_block_infor> {
    static const int value = static_cast<int>(Type::MMP);
};

template <>
struct mem_type_of<alloc::Nml_block_infor> {
    static const int value = static_cast<int>(Type::NML);
};

template <>
struct mem_type_of<alloc::Sml_block_infor> {
    static const int value = static_cast<int>(Type::SML);
};

static Manager globalMemPool;

bool Mmp_block_list::add_reference(const alloc::Mmp_block_infor& infor, Ref& mem_ref) {
    VSILOGD("[mem_pool]Add memory mapped block");

    auto found_at = std::find_if(mem_list_.begin(),
                            mem_list_.end(),
                            [&infor](const mem_list_type::value_type& kv){
                                if (kv.first == infor.address_) {
                                    VSILOGD("[mem_pool] shared memory map address");
                                    return true;
                                }
                                return false;
                            });

    if (found_at == mem_list_.end()) {
        auto insert = mem_list_.insert(std::make_pair(infor.address_, 0));
        if (insert.second) {
            found_at = insert.first;
        }
        else {
            VSILOGE("[mem_pool] insert operation failed");
            mem_ref = INVALID_REF;
            return false;
        }
    }

    // found_at is the position of required mmap block
    mem_ref.mem_type_ = Type::MMP;
    mem_ref.index_ = 0; //not used
    auto mem_end = (found_at->first)->length();
    if (infor.offset_ >= mem_end) {
        VSILOGE("[mem_pool]requested memory exceed current space");

        mem_ref = INVALID_REF;
        mem_list_.erase(found_at->first);
        return false;
    }

    mem_ref.offset_ = infor.offset_;

    auto request_end =  infor.len_ + infor.offset_;

    request_end < mem_end ?
        mem_ref.len_ = infor.len_ :
        mem_ref.len_ = mem_end - infor.offset_;

    mem_ref.address_ = (found_at->first)->data(mem_ref.offset_);
    mem_ref.src_ = found_at->first;
    ++ found_at->second;

    return true;
}

void Mmp_block_list::del_reference(const Ref& mem_ref) {
    VSILOGD("[mem_pool]Remove memory reference from MMP Block list");

    // auto idx = std::find(mem_list_.begin(), mem_list_.end(), mem_ref.address_);
    auto found_at = std::find_if(   mem_list_.begin(),
                                    mem_list_.end(),
                                    [&mem_ref](const mem_list_type::value_type& kv){
        return (kv.first == mem_ref.src_);
    });

    if (found_at != mem_list_.end()) {
        -- found_at->second;
        if (0 == found_at->second) {
            mem_list_.erase(found_at->first);
        }
    }
    else {
        VSILOGE("[mem_pool]Can not find given Memory Reference in Memory mapped pools");
    }

    VSILOGD("[mem_pool]$%d Memory mapped blocks tracked by system", mem_list_.size());
}

bool operator<(const Nml_block_list::Ro_segment& l, const Nml_block_list::Ro_segment& r) {
    return l.address_ < r.address_;
}

bool Nml_block_list::add_reference(const alloc::Nml_block_infor& infor, Ref& mem_ref) {
    VSILOGD("[mem_pool]Add normal memory block(size > 128)");

    auto found_at = std::find_if( mem_list_.begin(),
                                  mem_list_.end(),
                                  [&infor](const mem_list_type::value_type& kv){
        return infor.address_ == kv.first.address_;
    });

    if (found_at == mem_list_.end()) {
        Ro_segment seg;
        seg.address_ = infor.address_;
        seg.len_ = infor.len_;

        auto insert = mem_list_.insert(std::make_pair(seg, 0));
        if (insert.second) {
            found_at = insert.first;
        }
    }

    mem_ref.mem_type_ = Type::NML;
    mem_ref.address_ = infor.address_;
    mem_ref.index_ = 0; // not used
    mem_ref.len_ = infor.len_;
    mem_ref.offset_ = 0;
    mem_ref.src_ = infor.address_;
    ++ found_at->second;

    return true;
}

void Nml_block_list::del_reference(const Ref& mem_ref) {
    VSILOGD("[mem_pool]Delete normal memory reference");
    auto found_at = std::find_if(   mem_list_.begin(),
                                    mem_list_.end(),
                                    [&mem_ref](const mem_list_type::value_type& kv){
        return mem_ref.address_ == kv.first.address_;
    });

    if (found_at != mem_list_.end()) {
        -- found_at->second;
        if (0 == found_at->second) {
            mem_list_.erase(found_at->first);
        }
    }
    else {
        VSILOGE("[mem_pool]Can not find given Memory Refence in Normal RO memory block");
    }

    // trace usage information
    VSILOGD("[mem_pool]#%d normal memory blocks tracked by system", mem_list_.size());
}

Sml_block_list::Sml_block_list() {}

bool Sml_block_list::add_reference(const alloc::Sml_block_infor& infor, Ref& mem_ref) {
    VSILOGD("[mem_pool]Add small memory block to system");

    { // Increase reference count if source address already copied to internal memory
        auto found_at = std::find_if(ref_count_.begin(),
                                ref_count_.end(),
                                [&infor, &mem_ref](typename std::map<Ref, int>::value_type& item){
                                    if (item.first.src_ == infor.address_) {
                                        mem_ref = item.first;
                                        ++ item.second;
                                        VSILOGD("[mem_pool] Buffer@%p already"
                                                " copied to internal memory", infor.address_);
                                        return true;
                                    }
                                    return false;
                                });

        if (found_at != ref_count_.end()) {
            return true;
        }
    }
    mem_list_.emplace_back();
    mem_list_.back().resize(infor.len_);
    std::memcpy(mem_list_.back().data(), infor.address_, infor.len_);

    mem_ref.address_    = mem_list_.back().data();
    mem_ref.len_        = infor.len_;
    mem_ref.mem_type_   = Type::SML;
    mem_ref.index_      = mem_list_.size() -1;
    mem_ref.offset_     = 0;
    mem_ref.src_        = infor.address_;

    auto found_at = ref_count_.find(mem_ref);
    if (ref_count_.end() == found_at) {
        VSILOGD("[mem_pool] Add new ref_count");
        ref_count_.insert(std::make_pair(mem_ref, 0));
    }
    ++ ref_count_[mem_ref];

    VSILOGD("[mem_pool]Add %d(Bytes) small memory, ref_count_.size()=%d",
                mem_ref.len_, ref_count_.size());

    return true;
}

void Sml_block_list::del_reference(const Ref& mem_ref) {
    VSILOGD("[mem_pool]delete small block reference");
    int ref_cnt = --ref_count_[mem_ref];

    if (0 == ref_cnt) {
        VSILOGD("[mem_pool]release %dByte memory for sml_block", mem_ref.len_);;
        auto pos = std::find_if( mem_list_.begin(),
                                 mem_list_.end(),
                                 [&mem_ref](const std::vector<uint8_t>& item){
            return item.data() == mem_ref.address_;
        });

        mem_list_.erase(pos);
        mem_list_.shrink_to_fit();
        ref_count_.erase(mem_ref);
    }

    // usage information
    VSILOGD("[mem_pool]#%d small memory blocks shared total %dByte memory tracked by system",
            ref_count_.size(), mem_list_.size());
}

Manager& global_memory_pool() {
    return globalMemPool;
}

template <typename BlockAllocInfor>
shared_ref Manager::add_reference(const BlockAllocInfor& infor) {
    shared_ref ref_ptr(new Ref(), Ref_deleter());

    auto& block = std::get<mem_type_of<BlockAllocInfor>::value>(space_);

    if (!block.add_reference(infor, *ref_ptr)) {
        VSILOGW("Failed at add reference");
        ref_ptr.reset();
    }

    return ref_ptr;
}

shared_ref Manager::add_reference(const void* address, size_t len, bool is_output) {
    if (!is_output && len <= Sml_block_list::small_block_shreshold) {
        alloc::Sml_block_infor infor = {address, len};
        return add_reference(infor);
    }
    else {
        alloc::Nml_block_infor infor = {address, len};
        return add_reference(infor);
    }
}

shared_ref Manager::add_reference(const ovxlib::Memory* address, size_t offset, size_t len) {
    alloc::Mmp_block_infor infor = {address, offset, len};

    return add_reference(infor);
}

void Manager::del_reference(const Ref& mem_ref) {
    switch(mem_ref.mem_type_) {
        case Type::MMP:
        {
            auto& block = std::get<static_cast<int>(Type::MMP)>(space_);
            block.del_reference(mem_ref);
        }
        break;
        case Type::NML:
        {
            auto& block = std::get<static_cast<int>(Type::NML)>(space_);
            block.del_reference(mem_ref);
        }
        break;
        case Type::SML:
        {
            auto& block = std::get<static_cast<int>(Type::SML)>(space_);
            block.del_reference(mem_ref);
        }
        break;

        default:
            VSILOGE("[mem_pool]Should Never be here!!!");
    }
}

void Ref_deleter::operator()(Ref* mem_ref) const{
        global_memory_pool().del_reference(*mem_ref);

    delete mem_ref;
}

bool operator<(const Ref& left, const Ref& right) {
    return (left.address_ < right.address_);
}

}
