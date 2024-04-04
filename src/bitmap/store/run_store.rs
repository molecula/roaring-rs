use super::bitmap_store::{bit, key, BITMAP_LENGTH};
use super::{ArrayStore, BitmapStore};
use core::ops::{BitOrAssign, RangeInclusive};
use std::cmp;
use std::cmp::Ordering::{self, Equal, Greater, Less};

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct Interval {
    pub start: u16,
    pub end: u16,
}

fn cmp_index_interval(index: u16, iv: Interval) -> Ordering {
    if index < iv.start {
        Greater
    } else if index > iv.end {
        Less
    } else {
        Equal
    }
}

impl Interval {
    pub fn new(start: u16, end: u16) -> Interval {
        Interval { start, end }
    }

    pub fn run_len(&self) -> u64 {
        (self.end - self.start) as u64 + 1
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct RunStore {
    pub vec: Vec<Interval>,
}

impl RunStore {
    pub fn new() -> RunStore {
        RunStore { vec: vec![] }
    }

    pub fn with_capacity(capacity: usize) -> RunStore {
        RunStore { vec: Vec::with_capacity(capacity) }
    }

    ///
    /// Create a new SortedIntervalVec from a given vec
    /// It is up to the caller to ensure the vec is sorted and deduplicated
    ///
    /// # Panics
    ///
    /// When debug_assertions are enabled and the above invariants are not met
    #[inline]
    pub fn from_vec_unchecked(vec: Vec<Interval>) -> RunStore {
        RunStore { vec }
    }

    pub fn insert(&mut self, index: u16) -> bool {
        self.vec
            .binary_search_by(|iv| cmp_index_interval(index, *iv))
            .map_err(|loc| {
                // Value is beyond end of interval
                if loc > self.vec.len() - 1 && loc > 0 {
                    // The binary search indicated that the index value is
                    // beyond the existing intervals. We need to check if it is
                    // immediatly after the last interval end (and can therefore
                    // be incorporated into that existing interval), or if it
                    // requires a new interval.
                    if self.vec[loc - 1].end == index - 1 {
                        // Value immediately follows interval
                        self.vec[loc - 1].end += 1
                    } else {
                        // Otherwise create new standalone interval
                        self.vec.insert(loc, Interval::new(index, index));
                    }
                } else if self.vec[loc].end < index {
                    // If immediately follows this interval
                    if index == self.vec[loc].end + 1 {
                        if loc + 1 < self.vec.len() && index == self.vec[loc + 1].start {
                            // Merge with following interval
                            self.vec[loc].end = self.vec[loc + 1].end;
                            self.vec.remove(loc + 1);
                            return;
                        }
                        // Extend end of this interval by 1
                        self.vec[loc].end += 1
                    } else {
                        // Otherwise create new standalone interval
                        self.vec.insert(loc, Interval::new(index, index));
                    }
                } else if self.vec[loc].start == index + 1 {
                    // Value immediately precedes interval
                    if loc > 0 && self.vec[loc - 1].end == &index - 1 {
                        // Merge with preceding interval
                        self.vec[loc - 1].end = self.vec[loc].end;
                        self.vec.remove(loc);
                        return;
                    }
                    self.vec[loc].start -= 1;
                } else if loc > 0 && index - 1 == self.vec[loc - 1].end {
                    // Immediately follows the previous interval
                    self.vec[loc - 1].end += 1
                } else {
                    self.vec.insert(loc, Interval::new(index, index));
                }
            })
            .is_err()
    }

    pub fn remove(&mut self, index: u16) -> bool {
        self.vec
            .binary_search_by(|iv| cmp_index_interval(index, *iv))
            .map(|loc| {
                if index == self.vec[loc].start && index == self.vec[loc].end {
                    // Remove entire run if it only contains this value
                    self.vec.remove(loc);
                } else if index == self.vec[loc].end {
                    // Value is last in this interval
                    self.vec[loc].end = index - 1;
                } else if index == self.vec[loc].start {
                    // Value is first in this interval
                    self.vec[loc].start = index + 1;
                } else {
                    // Value lies inside the interval, we need to split it
                    // First construct a new interval with the right part
                    let new_interval = Interval::new(index + 1, self.vec[loc].end);
                    // Then shrink the current interval
                    self.vec[loc].end = index - 1;
                    // Then insert the new interval leaving gap where value was removed
                    self.vec.insert(loc + 1, new_interval);
                }
            })
            .is_ok()
    }

    pub fn remove_range(&mut self, range: RangeInclusive<u16>) -> u64 {
        let start = *range.start();
        let end = *range.end();

        let mut count = 0;
        let mut search_end = false;

        for iv in self.vec.iter_mut() {
            if !search_end && cmp_index_interval(start as u16, *iv) == Equal {
                count += Interval::new(iv.end, start as u16).run_len();
                iv.end = start as u16;
                search_end = true;
            }

            if search_end {
                // The end bound is non-inclusive therefore we must search for end - 1.
                match cmp_index_interval(end as u16 - 1, *iv) {
                    Less => {
                        // We invalidate the intervals that are contained in
                        // the start and end but doesn't touch the bounds.
                        count += iv.run_len();
                        *iv = Interval::new(u16::max_value(), 0);
                    }
                    Equal => {
                        // We shrink this interval by moving the start of it to be
                        // the end bound which is non-inclusive.
                        count += Interval::new(end as u16, iv.start).run_len();
                        iv.start = end as u16;
                    }
                    Greater => break,
                }
            }
        }

        // We invalidated the intervals to remove,
        // the start is greater than the end.
        self.vec.retain(|iv| iv.start <= iv.end);

        count
    }

    pub fn contains(&self, index: u16) -> bool {
        self.vec.binary_search_by(|iv| cmp_index_interval(index, *iv)).is_ok()
    }

    pub fn to_array_store(&self) -> ArrayStore {
        let arr = self.vec.iter().flat_map(|iv| iv.start..=iv.end).collect();
        ArrayStore::from_vec_unchecked(arr)
    }

    pub fn to_bitmap_store(&self) -> BitmapStore {
        let mut bits = Box::new([0; BITMAP_LENGTH]);
        let len = self.len();
        for iv in self.vec.iter() {
            for index in iv.start..=iv.end {
                bits[key(index)] |= 1 << bit(index);
            }
        }
        BitmapStore::from_unchecked(len, bits)
    }

    pub fn len(&self) -> u64 {
        self.vec.iter().map(|iv| iv.run_len() as u64).sum()
    }

    pub fn count_runs(&self) -> u64 {
        self.vec.len() as u64
    }

    pub fn min(&self) -> Option<u16> {
        Some(self.vec.first().unwrap().start)
    }

    pub fn max(&self) -> Option<u16> {
        Some(self.vec.last().unwrap().end)
    }

    pub fn iter(&self) -> RunIter {
        RunIter::new(self.vec.to_vec()) // TODO(tlt): this copy may be unnecessary
    }

    pub fn into_iter(self) -> RunIter {
        RunIter::new(self.vec.to_vec())
    }
}

pub struct RunIter {
    run: usize,
    offset: u64,
    intervals: Vec<Interval>,
}

impl RunIter {
    fn new(intervals: Vec<Interval>) -> RunIter {
        RunIter { run: 0, offset: 0, intervals }
    }

    fn move_next(&mut self) {
        self.offset += 1;
        if self.offset == self.intervals[self.run].run_len() {
            self.offset = 0;
            self.run += 1;
        }
    }
}

impl Iterator for RunIter {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        if self.run == self.intervals.len() {
            return None;
        }
        let result = self.intervals[self.run].start + self.offset as u16;
        self.move_next();
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        panic!("Should never be called (roaring::Iter caches the size_hint itself)")
    }
}

impl BitOrAssign<&Self> for RunStore {
    fn bitor_assign(&mut self, intervals2: &Self) {
        let mut merged = Vec::new();

        let (mut i1, mut i2) = (self.vec.iter(), intervals2.vec.iter());
        let (mut iv1, mut iv2) = (i1.next(), i2.next());
        loop {
            // Iterate over two iterators and return the lowest value at each step.
            let iv = match (iv1, iv2) {
                (None, None) => break,
                (Some(v1), None) => {
                    iv1 = i1.next();
                    v1
                }
                (None, Some(v2)) => {
                    iv2 = i2.next();
                    v2
                }
                (Some(v1), Some(v2)) => match v1.start.cmp(&v2.start) {
                    Equal => {
                        iv1 = i1.next();
                        iv2 = i2.next();
                        v1
                    }
                    Less => {
                        iv1 = i1.next();
                        v1
                    }
                    Greater => {
                        iv2 = i2.next();
                        v2
                    }
                },
            };

            match merged.last_mut() {
                // If the list of merged intervals is empty, append the interval.
                None => merged.push(*iv),
                Some(last) => {
                    if last.end < iv.start {
                        // If the interval does not overlap with the previous, append it.
                        merged.push(*iv);
                    } else {
                        // If there is overlap, so we merge the current and previous intervals.
                        last.end = cmp::max(last.end, iv.end);
                    }
                }
            }
        }

        self.vec = merged;
    }
}

impl BitOrAssign<&ArrayStore> for RunStore {
    fn bitor_assign(&mut self, rhs: &ArrayStore) {
        for i in rhs.iter() {
            self.insert(*i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::store::Store;

    fn into_vec(s: Store) -> Vec<Interval> {
        match s {
            Store::Array(_vec) => panic!("array not implemented"),
            Store::Bitmap(_bits) => panic!("bitmap not implemented"),
            Store::Run(vec) => vec.vec,
        }
    }

    #[test]
    fn test_run_insert_before_with_no_gap() {
        let mut store = Store::Run(RunStore::from_vec_unchecked(vec![Interval::new(10, 20)]));

        let new = store.insert(9);
        assert_eq!(new, true);

        assert_eq!(into_vec(store), vec![Interval::new(9, 20)]);
    }

    #[test]
    fn test_run_insert_before_with_gap() {
        let mut store = Store::Run(RunStore::from_vec_unchecked(vec![Interval::new(10, 20)]));

        let new = store.insert(8);
        assert_eq!(new, true);

        assert_eq!(into_vec(store), vec![Interval::new(8, 8), Interval::new(10, 20)]);
    }

    #[test]
    fn test_run_insert_after_with_no_gap() {
        let mut store = Store::Run(RunStore::from_vec_unchecked(vec![Interval::new(10, 20)]));

        let new = store.insert(21);
        assert_eq!(new, true);

        assert_eq!(into_vec(store), vec![Interval::new(10, 21)]);
    }

    #[test]
    fn test_run_insert_after_with_gap() {
        let mut store = Store::Run(RunStore::from_vec_unchecked(vec![Interval::new(10, 20)]));

        let new = store.insert(22);
        assert_eq!(new, true);

        assert_eq!(into_vec(store), vec![Interval::new(10, 20), Interval::new(22, 22)]);
    }

    #[test]
    fn test_run_insert_between_with_gap() {
        let mut store = Store::Run(RunStore::from_vec_unchecked(vec![
            Interval::new(10, 20),
            Interval::new(30, 40),
        ]));

        let new = store.insert(25);
        assert_eq!(new, true);

        assert_eq!(
            into_vec(store),
            vec![Interval::new(10, 20), Interval::new(25, 25), Interval::new(30, 40)]
        );
    }

    #[test]
    fn test_run_insert_between_with_no_gap() {
        let mut store = Store::Run(RunStore::from_vec_unchecked(vec![
            Interval::new(10, 20),
            Interval::new(22, 40),
        ]));

        let new = store.insert(21);
        assert_eq!(new, true);

        assert_eq!(into_vec(store), vec![Interval::new(10, 40)]);
    }
}
