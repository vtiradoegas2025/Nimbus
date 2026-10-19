#pragma once

#include "diagnostics/budget_level.hpp"
#include "util/log.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace tmv
{
class LevelBudgetCsv;

// Scientific rows are published on the orchestration thread after its outer
// state transaction commits. Nested transactions remain part of that commit.
class BudgetPublicationTransaction
{
public:
    BudgetPublicationTransaction() : parent_(active_) { active_ = this; }
    ~BudgetPublicationTransaction() { active_ = parent_; }
    BudgetPublicationTransaction(const BudgetPublicationTransaction&) = delete;
    BudgetPublicationTransaction& operator=(const BudgetPublicationTransaction&) = delete;
    void commit();

private:
    friend class LevelBudgetCsv;
    struct Pending
    {
        LevelBudgetCsv* publisher;
        BudgetLevel level;
        std::mutex* mutex;
        std::uint64_t generation;
        std::ostringstream rows;
    };
    std::ostream* stream(LevelBudgetCsv& publisher, BudgetLevel level, std::mutex& mutex);
    inline static thread_local BudgetPublicationTransaction* active_ = nullptr;
    BudgetPublicationTransaction* parent_;
    std::vector<std::unique_ptr<Pending>> pending_;
};

// Access is serialized by each budget publisher's mutex. Child histories
// describe the child volume, not an additive correction to the parent budget.
class LevelBudgetCsv
{
public:
    bool open(const std::string& path, const std::string& header, int precision)
    {
        close();
        header_ = header;
        precision_ = precision;
        child_path_ = std::filesystem::path(path).parent_path()
            / "nests" / "child_0000" / std::filesystem::path(path).filename();
        root_.open(path, std::ios::out | std::ios::trunc);
        if (!root_) return false;
        root_ << header_;
        root_.precision(precision_);
        return true;
    }

    std::ostream* stream(BudgetLevel level, std::mutex& mutex)
    {
        if (!root_.is_open()) return nullptr;
        if (auto* transaction = BudgetPublicationTransaction::active_)
            return transaction->stream(*this, level, mutex);
        return direct_stream(level);
    }

    void close()
    {
        for (auto* stream : {&root_, &child_})
        {
            if (stream->is_open()) stream->close();
            stream->clear();
        }
        child_attempted_ = false;
        ++generation_;
    }

private:
    friend class BudgetPublicationTransaction;
    std::ofstream* direct_stream(BudgetLevel level)
    {
        if (!root_.is_open()) return nullptr;
        if (level == BudgetLevel::Root) return &root_;
        if (!child_attempted_)
        {
            child_attempted_ = true;
            std::error_code error;
            std::filesystem::create_directories(child_path_.parent_path(), error);
            if (!error) child_.open(child_path_, std::ios::out | std::ios::trunc);
            if (!child_.is_open())
            {
                log_warn("[BUDGET CSV] failed to open child history '", child_path_.string(), "'");
                return nullptr;
            }
            child_ << header_;
            child_.precision(precision_);
        }
        return child_.is_open() ? &child_ : nullptr;
    }

    std::ofstream root_;
    std::ofstream child_;
    std::filesystem::path child_path_;
    std::string header_;
    int precision_ = 17;
    bool child_attempted_ = false;
    std::uint64_t generation_ = 0;
};

inline std::ostream* BudgetPublicationTransaction::stream(
    LevelBudgetCsv& publisher, BudgetLevel level, std::mutex& mutex)
{
    for (auto entry = pending_.rbegin(); entry != pending_.rend(); ++entry)
        if ((*entry)->publisher == &publisher && (*entry)->level == level &&
            (*entry)->generation == publisher.generation_)
            return &(*entry)->rows;
    auto entry = std::make_unique<Pending>();
    entry->publisher = &publisher;
    entry->level = level;
    entry->mutex = &mutex;
    entry->generation = publisher.generation_;
    entry->rows.copyfmt(publisher.root_);
    auto* result = &entry->rows;
    pending_.push_back(std::move(entry));
    return result;
}

inline void BudgetPublicationTransaction::commit()
{
    if (parent_)
    {
        parent_->pending_.reserve(parent_->pending_.size() + pending_.size());
        for (auto& entry : pending_) parent_->pending_.push_back(std::move(entry));
    }
    else
    {
        std::vector<std::string> rows;
        rows.reserve(pending_.size());
        for (const auto& entry : pending_) rows.push_back(entry->rows.str());
        for (std::size_t index = 0; index < pending_.size(); ++index)
        {
            const auto& entry = *pending_[index];
            std::lock_guard<std::mutex> lock(*entry.mutex);
            if (entry.generation == entry.publisher->generation_)
                if (auto* destination = entry.publisher->direct_stream(entry.level))
                    *destination << rows[index];
        }
    }
    pending_.clear();
}
}
