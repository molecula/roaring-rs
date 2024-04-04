use core::fmt;
use core::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, RangeInclusive, Sub, SubAssign,
};

use alloc::vec::Vec;

pub use super::store::Store;
use super::store::{self};
use super::util;

// ARRAY_LIMIT is the maximum number of array elements in an array container.
// Normally, its value would be 4096, but since our page size is 8152 bytes, we
// set it to a lower value to avoid overflowing a single page.
pub const ARRAY_LIMIT: u64 = 4076;

// RUN_MAX_SIZE is the maximum number of runs in a run container. Normally, its
// value would be 2048, but since our page size is 8152 bytes, we set it to a
// lower value to avoid overflowing a single page.
pub const RUN_MAX_SIZE: u64 = 2038;

#[derive(PartialEq, Clone)]
pub struct Container {
    pub key: u16,
    pub store: Store,
}

pub struct Iter<'a> {
    pub key: u16,
    inner: store::Iter<'a>,
}

impl Container {
    pub fn new(key: u16) -> Container {
        Container { key, store: Store::new() }
    }

    pub fn full(key: u16) -> Container {
        Container { key, store: Store::full() }
    }
}

impl Container {
    pub fn len(&self) -> u64 {
        self.store.len()
    }

    pub fn insert(&mut self, index: u16) -> bool {
        if self.store.insert(index) {
            self.ensure_correct_store();
            true
        } else {
            false
        }
    }

    pub fn insert_range(&mut self, range: RangeInclusive<u16>) -> u64 {
        // If inserting the range will make this a bitmap by itself, do it now
        if range.len() as u64 > ARRAY_LIMIT {
            if let Store::Array(arr) = &self.store {
                self.store = Store::Bitmap(arr.to_bitmap_store());
            }
        }
        let inserted = self.store.insert_range(range);
        self.ensure_correct_store();
        inserted
    }

    /// Pushes `index` at the end of the container only if `index` is the new max.
    ///
    /// Returns whether the `index` was effectively pushed.
    pub fn push(&mut self, index: u16) -> bool {
        if self.store.push(index) {
            self.ensure_correct_store();
            true
        } else {
            false
        }
    }

    ///
    /// Pushes `index` at the end of the container.
    /// It is up to the caller to have validated index > self.max()
    ///
    /// # Panics
    ///
    /// If debug_assertions enabled and index is > self.max()
    pub(crate) fn push_unchecked(&mut self, index: u16) {
        self.store.push_unchecked(index);
        self.ensure_correct_store();
    }

    pub fn remove(&mut self, index: u16) -> bool {
        if self.store.remove(index) {
            self.ensure_correct_store();
            true
        } else {
            false
        }
    }

    pub fn remove_range(&mut self, range: RangeInclusive<u16>) -> u64 {
        let result = self.store.remove_range(range);
        self.ensure_correct_store();
        result
    }

    pub fn remove_smallest(&mut self, n: u64) {
        match &self.store {
            Store::Bitmap(bits) => {
                if bits.len() - n <= ARRAY_LIMIT {
                    let mut replace_array = Vec::with_capacity((bits.len() - n) as usize);
                    replace_array.extend(bits.iter().skip(n as usize));
                    self.store = Store::Array(store::ArrayStore::from_vec_unchecked(replace_array));
                } else {
                    self.store.remove_smallest(n)
                }
            }
            Store::Array(_) => self.store.remove_smallest(n),
            Store::Run(_) => panic!("run not implemented"),
        };
    }

    pub fn remove_biggest(&mut self, n: u64) {
        match &self.store {
            Store::Bitmap(bits) => {
                if bits.len() - n <= ARRAY_LIMIT {
                    let mut replace_array = Vec::with_capacity((bits.len() - n) as usize);
                    replace_array.extend(bits.iter().take((bits.len() - n) as usize));
                    self.store = Store::Array(store::ArrayStore::from_vec_unchecked(replace_array));
                } else {
                    self.store.remove_biggest(n)
                }
            }
            Store::Array(_) => self.store.remove_biggest(n),
            Store::Run(_) => panic!("run not implemented"),
        };
    }

    pub fn contains(&self, index: u16) -> bool {
        self.store.contains(index)
    }

    pub fn contains_range(&self, range: RangeInclusive<u16>) -> bool {
        self.store.contains_range(range)
    }

    pub fn is_full(&self) -> bool {
        self.store.is_full()
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.store.is_disjoint(&other.store)
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.store.is_subset(&other.store)
    }

    pub fn intersection_len(&self, other: &Self) -> u64 {
        self.store.intersection_len(&other.store)
    }

    pub fn min(&self) -> Option<u16> {
        self.store.min()
    }

    pub fn max(&self) -> Option<u16> {
        self.store.max()
    }

    pub fn rank(&self, index: u16) -> u64 {
        self.store.rank(index)
    }

    /// ensure_correct_store ensures that the state of the store is valid and
    /// will not overflow. For example, it ensures that an array store or a run
    /// store are not too large. It does do one optimization: it will convert a
    /// bitmap store to an array store when possible. With that said, full
    /// optimization should be done with the `optimize()` method.
    ///
    /// This method returns true if the store was changed to a different type.
    pub(crate) fn ensure_correct_store(&mut self) -> bool {
        match &self.store {
            Store::Bitmap(ref bits) => {
                if bits.len() <= ARRAY_LIMIT {
                    self.store = Store::Array(bits.to_array_store());
                    return true;
                }
            }
            Store::Array(ref vec) => {
                if vec.len() > ARRAY_LIMIT {
                    self.store = Store::Bitmap(vec.to_bitmap_store());
                    return true;
                }
            }
            Store::Run(ref intervals) => {
                if intervals.len() > RUN_MAX_SIZE {
                    self.store = Store::Bitmap(intervals.to_bitmap_store());
                    return true;
                }
            }
        }
        false
    }

    /// optimize is effectively a wrapper around `ensure_correct_store()`, but
    /// in addition to ensuring that the store is valid, it will convert a store
    /// to a RunStore when applicable. This is a separate function under the
    /// assumption that doing the `num_runs()` call on a store for most
    /// operations would not be very efficient. If that turns out not to be the
    /// case, we could collapse `optimize()` into the `ensure_correct_store()`
    /// method.
    pub fn optimize(&mut self) -> bool {
        match self.store {
            Store::Array(..) | Store::Bitmap(..) => {
                let num_runs = self.store.count_runs();
                if num_runs <= RUN_MAX_SIZE && num_runs <= self.len() / 2 {
                    self.store = self.store.to_run();
                    true
                } else {
                    self.ensure_correct_store()
                }
            }
            Store::Run(..) => self.ensure_correct_store(),
        }
    }
}

impl BitOr<&Container> for &Container {
    type Output = Container;

    fn bitor(self, rhs: &Container) -> Container {
        let store = BitOr::bitor(&self.store, &rhs.store);
        let mut container = Container { key: self.key, store };
        container.ensure_correct_store();
        container
    }
}

impl BitOrAssign<Container> for Container {
    fn bitor_assign(&mut self, rhs: Container) {
        BitOrAssign::bitor_assign(&mut self.store, rhs.store);
        self.ensure_correct_store();
    }
}

impl BitOrAssign<&Container> for Container {
    fn bitor_assign(&mut self, rhs: &Container) {
        BitOrAssign::bitor_assign(&mut self.store, &rhs.store);
        self.ensure_correct_store();
    }
}

impl BitAnd<&Container> for &Container {
    type Output = Container;

    fn bitand(self, rhs: &Container) -> Container {
        let store = BitAnd::bitand(&self.store, &rhs.store);
        let mut container = Container { key: self.key, store };
        container.ensure_correct_store();
        container
    }
}

impl BitAndAssign<Container> for Container {
    fn bitand_assign(&mut self, rhs: Container) {
        BitAndAssign::bitand_assign(&mut self.store, rhs.store);
        self.ensure_correct_store();
    }
}

impl BitAndAssign<&Container> for Container {
    fn bitand_assign(&mut self, rhs: &Container) {
        BitAndAssign::bitand_assign(&mut self.store, &rhs.store);
        self.ensure_correct_store();
    }
}

impl Sub<&Container> for &Container {
    type Output = Container;

    fn sub(self, rhs: &Container) -> Container {
        let store = Sub::sub(&self.store, &rhs.store);
        let mut container = Container { key: self.key, store };
        container.ensure_correct_store();
        container
    }
}

impl SubAssign<&Container> for Container {
    fn sub_assign(&mut self, rhs: &Container) {
        SubAssign::sub_assign(&mut self.store, &rhs.store);
        self.ensure_correct_store();
    }
}

impl BitXor<&Container> for &Container {
    type Output = Container;

    fn bitxor(self, rhs: &Container) -> Container {
        let store = BitXor::bitxor(&self.store, &rhs.store);
        let mut container = Container { key: self.key, store };
        container.ensure_correct_store();
        container
    }
}

impl BitXorAssign<Container> for Container {
    fn bitxor_assign(&mut self, rhs: Container) {
        BitXorAssign::bitxor_assign(&mut self.store, rhs.store);
        self.ensure_correct_store();
    }
}

impl BitXorAssign<&Container> for Container {
    fn bitxor_assign(&mut self, rhs: &Container) {
        BitXorAssign::bitxor_assign(&mut self.store, &rhs.store);
        self.ensure_correct_store();
    }
}

impl<'a> IntoIterator for &'a Container {
    type Item = u32;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        let store: &Store = &self.store;
        Iter { key: self.key, inner: store.into_iter() }
    }
}

impl IntoIterator for Container {
    type Item = u32;
    type IntoIter = Iter<'static>;

    fn into_iter(self) -> Iter<'static> {
        Iter { key: self.key, inner: self.store.into_iter() }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        self.inner.next().map(|i| util::join(self.key, i))
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|i| util::join(self.key, i))
    }
}

impl fmt::Debug for Container {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        format!("Container<{:?} @ {:?}>", self.len(), self.key).fmt(formatter)
    }
}
