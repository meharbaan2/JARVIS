using System.Diagnostics;
using System.Text.Json;
using FolderGuardian.Core;

internal static class Program
{
    private const int MaxLogEntries = 200;

    private static async Task<int> Main(string[] args)
    {
        var options = ParseArgs(args);
        if (options.Command is "help" or null)
        {
            WriteJson(new Dictionary<string, object?>
            {
                ["status"] = "success",
                ["commands"] = new[] { "encrypt", "decrypt" },
                ["usage"] = "FolderGuardianBridge encrypt --folder <path> --json",
            });
            return 0;
        }

        if (options.Command is not ("encrypt" or "decrypt"))
        {
            return WriteFailure($"Unknown command: {options.Command}", options.Command ?? "");
        }

        if (string.IsNullOrWhiteSpace(options.FolderPath))
        {
            return WriteFailure("--folder is required", options.Command);
        }

        string fullPath;
        try
        {
            fullPath = Path.GetFullPath(options.FolderPath);
        }
        catch (Exception ex)
        {
            return WriteFailure($"Folder path could not be resolved: {ex.Message}", options.Command, options.FolderPath);
        }

        if (!Directory.Exists(fullPath))
        {
            return WriteFailure($"Folder does not exist: {fullPath}", options.Command, fullPath);
        }

        var log = new List<string>();
        int logOverflow = 0;
        void Logger(string message)
        {
            if (log.Count < MaxLogEntries)
            {
                log.Add(message);
            }
            else
            {
                logOverflow++;
            }
        }

        FolderOperationProgress? latestProgress = null;
        void Progress(FolderOperationProgress progress)
        {
            latestProgress = progress;
        }

        var stopwatch = Stopwatch.StartNew();
        try
        {
            FolderOperationSummary summary = options.Command == "encrypt"
                ? await FolderEncryptor.EncryptFolderAsync(fullPath, Logger, Progress)
                : await FolderEncryptor.DecryptFolderAsync(fullPath, Logger, Progress);
            stopwatch.Stop();

            string status = summary.FailedCount > 0 ? "partial" : "success";
            WriteJson(new Dictionary<string, object?>
            {
                ["status"] = status,
                ["action"] = options.Command,
                ["folder_path"] = fullPath,
                ["summary"] = SummaryToJson(summary),
                ["latest_progress"] = ProgressToJson(latestProgress),
                ["log"] = log,
                ["log_overflow_count"] = logOverflow,
                ["key_location"] = EncryptionHelper.KeyLocationDescription,
                ["elapsed_ms"] = (long)stopwatch.Elapsed.TotalMilliseconds,
            });
            return 0;
        }
        catch (OperationCanceledException ex)
        {
            stopwatch.Stop();
            WriteJson(new Dictionary<string, object?>
            {
                ["status"] = "cancelled",
                ["action"] = options.Command,
                ["folder_path"] = fullPath,
                ["error_message"] = ex.Message,
                ["log"] = log,
                ["log_overflow_count"] = logOverflow,
                ["elapsed_ms"] = (long)stopwatch.Elapsed.TotalMilliseconds,
            });
            return 2;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            WriteJson(new Dictionary<string, object?>
            {
                ["status"] = "failed",
                ["action"] = options.Command,
                ["folder_path"] = fullPath,
                ["error_message"] = ex.Message,
                ["exception_type"] = ex.GetType().FullName,
                ["log"] = log,
                ["log_overflow_count"] = logOverflow,
                ["elapsed_ms"] = (long)stopwatch.Elapsed.TotalMilliseconds,
            });
            return 1;
        }
    }

    private static Dictionary<string, object?> SummaryToJson(FolderOperationSummary summary)
    {
        return new Dictionary<string, object?>
        {
            ["processed_count"] = summary.ProcessedCount,
            ["skipped_count"] = summary.SkippedCount,
            ["failed_count"] = summary.FailedCount,
            ["total_count"] = summary.TotalCount,
            ["duration_ms"] = (long)summary.Duration.TotalMilliseconds,
        };
    }

    private static Dictionary<string, object?>? ProgressToJson(FolderOperationProgress? progress)
    {
        if (progress is null)
        {
            return null;
        }

        return new Dictionary<string, object?>
        {
            ["phase"] = progress.Phase,
            ["current_item"] = progress.CurrentItem,
            ["completed_count"] = progress.CompletedCount,
            ["total_count"] = progress.TotalCount,
            ["failed_count"] = progress.FailedCount,
            ["estimated_remaining_ms"] = progress.EstimatedRemaining is null
                ? null
                : (long?)progress.EstimatedRemaining.Value.TotalMilliseconds,
        };
    }

    private static BridgeOptions ParseArgs(string[] args)
    {
        if (args.Length == 0)
        {
            return new BridgeOptions("help", null);
        }

        string? command = args[0].Trim().ToLowerInvariant();
        string? folderPath = null;
        for (int index = 1; index < args.Length; index++)
        {
            string arg = args[index];
            if (arg.Equals("--folder", StringComparison.OrdinalIgnoreCase) && index + 1 < args.Length)
            {
                folderPath = args[++index];
            }
        }

        return new BridgeOptions(command, folderPath);
    }

    private static int WriteFailure(string message, string action, string? folderPath = null)
    {
        WriteJson(new Dictionary<string, object?>
        {
            ["status"] = "failed",
            ["action"] = action,
            ["folder_path"] = folderPath,
            ["error_message"] = message,
        });
        return 1;
    }

    private static void WriteJson(object payload)
    {
        var options = new JsonSerializerOptions { WriteIndented = true };
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine(JsonSerializer.Serialize(payload, options));
    }

    private sealed record BridgeOptions(string? Command, string? FolderPath);
}
